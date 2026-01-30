# -*- coding: utf-8 -*-
"""
MCM 2026 Problem B: Q1 & Q2 Integrated Supply Chain Model (V9.8.1 Final Patch)
逻辑架构：Hybrid Push-Pull Supply Chain + Full Analysis Suite

本次补丁修复点（对齐建模手要求 & 评委可解释性）：
1) Construction口径修正：cum_M 表示“完成的基建吨位”，由 B/K 配比短板决定（不再材料吨直接累加）
2) Push闭环修正：TR/SE push 上限绑定 remain_goal，避免收尾阶段堆库导致NPV异常
3) Pull增强：B/K 的 gap target 增加当年基建需求项（更自适应；push 仍保留）
4) Efficiency：保持 SE_INV_EFF（投资有效转化率），避免无损升级质疑
5) 三合一：参数扫描 + 敏感性 + 鲁棒性

"""

import math
import csv
import random
from dataclasses import dataclass, field
from typing import Dict, Callable, List

# ==============================================================================
# 0. 全局参数 (物理/经济/供应链)
# ==============================================================================
# 物理常数
Re, h_apex = 6.37e6, 1e8
r_apex = Re + h_apex
mu_e, omega, g0 = 3.986e14, 7.292e-5, 9.81

# 任务目标
M_TOTAL_GOAL = 1e8
T_MAX = 65
P0 = 1e5

# Logistic Parameters (S-Curve)
LOGISTIC_DEMAND_K = 0.2
LOGISTIC_DEMAND_MID = 2065
LOGISTIC_BUILD_K = 0.25
LOGISTIC_BUILD_MID = 2060
MAX_ANNUAL_BUILD = 5e6

# Supply Chain
RHO_SUS = {'U': 0.02, 'K': 0.18, 'B': 0.80}
RHO_BUILD = {'U': 0.00, 'K': 0.10, 'B': 0.90}

GOODS_CFG = {
    'U': {'z': 2.58, 'sigma': 200,  'p': 2e6},
    'K': {'z': 1.96, 'sigma': 800,  'p': 5e5},
    'B': {'z': 1.64, 'sigma': 2000, 'p': 1e5},
}
H_HOLD = 500

MODES = {
    'D':  {'L': 0.25, 'CostF': 1.00, 'CarbF': 1.00},
    'B':  {'L': 0.50, 'CostF': 0.85, 'CarbF': 0.70},
    'SE': {'L': 0.10, 'CostF': 'Calc', 'CarbF': 'Calc'}
}
W_TIME = 1e5
W_CARB = 5.0
ROUTING = {'U': ['D'], 'K': ['D', 'B'], 'B': ['D', 'B', 'SE']}

# SE Parameters
SE_CAP_PER_PORT = 179000.0
SE_PORTS_DEFAULT = [{'id': 1, 'on': 1}, {'id': 2, 'on': 1}, {'id': 3, 'on': 1}]
PORTS_DEFAULT = len(SE_PORTS_DEFAULT)
CSE_THEORY = SE_CAP_PER_PORT * PORTS_DEFAULT
SE_INV_EFF = 0.8  # 投资转化效率（有效投入比例）

# Physics & Cost
W_JKG_SAFE_MAX = 1.8e8
E_PRICE = 0.15
ENERGY_EFF = 0.9
C_SE_OPEX_BASE = 1e4
eSE_base = 20.0
eSE_energy = 0.05

dV_ferry = 1800.0
Isp_ferry = 350.0
FUEL_PRICE = 8e4
FUEL_RATIO = math.exp(dV_ferry / (Isp_ferry * g0)) - 1

cTR_Base = 1.5e6
cTR_Apex_Ratio = 0.65
eTR_Base = 5000.0
LAT_PENALTY_WEIGHT = 0.3

R_DISC = 0.05
CARBON_TAX = 50.0

KA, KETA, KS = 1.2e-7, 1.2e-7, 1.0e-7

# ==============================================================================
# 1. 站点网络
# ==============================================================================
RAW_SITES = [
    ("Alcantara", -2.3, 150000), ("Kourou", 5.2, 200000), ("Wenchang", 19.6, 250000),
    ("Boca Chica", 26.0, 500000), ("Canaveral", 28.5, 450000), ("Tanegashima", 30.4, 80000),
    ("Vandenberg", 34.8, 100000), ("Taiyuan", 38.8, 120000), ("Jiuquan", 40.9, 150000),
    ("Baikonur", 46.0, 200000), ("Vostochny", 51.9, 100000), ("Plesetsk", 62.9, 80000)
]

@dataclass
class TRSite:
    name: str
    lat: float
    base_cap: float
    k_lat: float = field(init=False)
    current_used: float = 0.0

    def __post_init__(self):
        cos_val = math.cos(math.radians(self.lat))
        # Soft Latitude Penalty: k = 1 - w*(1-cos(lat))
        self.k_lat = 1.0 - LAT_PENALTY_WEIGHT * (1.0 - cos_val)

    def reset_usage(self):
        self.current_used = 0.0

    def get_effective_cost(self, base_cost: float, mode_factor: float) -> float:
        return (base_cost * mode_factor) / max(self.k_lat, 1e-9)

    def get_effective_carb(self, base_carb: float, mode_factor: float) -> float:
        return (base_carb * mode_factor) / max(self.k_lat, 1e-9)

    def get_avail_cap(self, t, age_fn, fail_fn) -> float:
        total = self.base_cap * self.k_lat * age_fn(t) * fail_fn(t)
        return max(0.0, total - self.current_used)

SITE_NETWORK = [TRSite(n, l, c) for n, l, c in RAW_SITES]
SITE_NETWORK.sort(key=lambda s: s.get_effective_cost(cTR_Base, 1.0))

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def unit_work_J_per_kg():
    return mu_e * (1 / Re - 1 / r_apex) - 0.5 * omega**2 * (r_apex**2 - Re**2)

def calc_se_opex(ovr=None):
    price = ovr.get('E_PRICE', E_PRICE) if ovr else E_PRICE
    eff = ovr.get('ENERGY_EFF', ENERGY_EFF) if ovr else ENERGY_EFF
    base = ovr.get('C_SE_OPEX_BASE', C_SE_OPEX_BASE) if ovr else C_SE_OPEX_BASE

    W = max(0.0, min(unit_work_J_per_kg(), W_JKG_SAFE_MAX))
    kWh_ton = (W / 3.6e6) * 1000.0
    cost = (kWh_ton * price) / max(eff, 1e-9) + base
    carb = eSE_base + kWh_ton * eSE_energy
    return cost, carb

def logistic_curve(t, start_val, end_val, midpoint, k):
    return end_val + (start_val - end_val) / (1 + math.exp(k * (t - midpoint)))

def split_arrival(amount, lead_time):
    if amount <= 0:
        return []
    n = int(math.floor(lead_time))
    tau = lead_time - n
    res = [(n, amount * (1 - tau))]
    if tau > 1e-6:
        res.append((n + 1, amount * tau))
    return res

def get_pipeline_sum(pipeline, c, t, horizon=2):
    total = 0.0
    for i in range(horizon + 1):
        total += pipeline[c].get(t + i, 0.0)
    return total

def update_se_state(st, I_total):
    A, eta, s = st['A'], st['eta'], st['s']
    nA = min(A + (1 - A) * KA * I_total, s + (1 - s) * KS * I_total)
    nEta = eta + (1 - eta) * KETA * I_total
    nS = s + (1 - s) * KS * I_total
    return {'A': min(nA, 1.0), 'eta': min(nEta, 1.0), 's': min(nS, 1.0)}

# ==============================================================================
# 3. 仿真配置
# ==============================================================================
@dataclass
class SimConfig:
    is_perfect_Q1: bool = True
    TR_age_fn: Callable = lambda t: 1.0
    TR_fail_fn: Callable = lambda t: 1.0
    SE_ports_count_fn: Callable = lambda t: PORTS_DEFAULT

def get_config_Q1():
    return SimConfig(is_perfect_Q1=True)

def get_config_Q2():
    return SimConfig(
        is_perfect_Q1=False,
        TR_age_fn=lambda t: max(0.6, 1.0 - 0.005 * (t - 2050)),
        TR_fail_fn=lambda t: 0.95
    )

# ==============================================================================
# 4. 核心仿真引擎 (The Engine)
# ==============================================================================
def simulate_scenario(alpha: float, beta: float, cfg: SimConfig, overrides: Dict = None):
    ovr = overrides if overrides else {}
    # Overrides
    p_cTR = ovr.get('cTR_Base', cTR_Base)
    p_eTR = ovr.get('eTR_Base', eTR_Base)
    p_fuel = ovr.get('FUEL_PRICE', FUEL_PRICE)
    p_tax = ovr.get('CARBON_TAX', CARBON_TAX)
    p_disc = ovr.get('R_DISC', R_DISC)
    p_M_total = ovr.get('M_TOTAL_GOAL', M_TOTAL_GOAL)
    p_inv_eff = ovr.get('SE_INV_EFF', SE_INV_EFF)

    # Init state
    if cfg.is_perfect_Q1:
        state = {'A': 1.0, 'eta': 1.0, 's': 1.0}
    else:
        state = {'A': 0.85, 'eta': 0.7, 's': 0.4}

    cSE_op, eSE_op = calc_se_opex(ovr)

    # Safety Stock
    SS = {}
    for c in GOODS_CFG:
        modes = ROUTING[c]
        avg_L = sum(MODES[m]['L'] for m in modes) / len(modes)
        SS[c] = GOODS_CFG[c]['z'] * GOODS_CFG[c]['sigma'] * math.sqrt(avg_L)

    # Inventory & pipeline
    inv = {c: 1.2 * SS[c] for c in GOODS_CFG}
    pipeline = {c: {} for c in GOODS_CFG}

    cum_M, cum_NPV, cum_C = 0.0, 0.0, 0.0
    cum_stockout = 0.0
    history = []

    finish_year = 2050 + T_MAX
    is_finished = False

    for t in range(2050, 2050 + T_MAX + 1):
        for s in SITE_NETWORK:
            s.reset_usage()

        # --- A. 需求与基建能力 ---
        curr_m_pc = logistic_curve(t, 2.0, 0.5, LOGISTIC_DEMAND_MID, LOGISTIC_DEMAND_K)
        D_sus_tot = P0 * curr_m_pc

        build_cap_limit = logistic_curve(
            t, MAX_ANNUAL_BUILD * 0.1, MAX_ANNUAL_BUILD, LOGISTIC_BUILD_MID, LOGISTIC_BUILD_K
        )

        remain_goal = max(0.0, p_M_total - cum_M)
        annual_build_target = min(build_cap_limit, remain_goal)

        # --- Pull Gap: sus + SS (+ build need for K/B) ---
        gaps = {}
        for c in GOODS_CFG:
            sus_req = D_sus_tot * RHO_SUS[c]
            IP = inv[c] + get_pipeline_sum(pipeline, c, t, horizon=1)

            # [增强] B/K 的补货目标考虑当年基建需要（更自适应）
            build_need_c = 0.0
            if c in ('B', 'K'):
                build_need_c = annual_build_target * RHO_BUILD[c]

            target = sus_req + SS[c] + build_need_c
            gaps[c] = max(0.0, target - IP)

        # --- B. 到货与消耗 & 基建转化 ---
        arrived = {c: pipeline[c].pop(t, 0.0) for c in GOODS_CFG}
        stockout_cost = 0.0

        # 1) 先入库
        for c in GOODS_CFG:
            inv[c] += arrived[c]

        # 2) 先扣维持需求（保命优先）
        for c in GOODS_CFG:
            sus_req = D_sus_tot * RHO_SUS[c]
            if inv[c] >= sus_req:
                inv[c] -= sus_req
            else:
                shortage = sus_req - inv[c]
                stockout_cost += shortage * GOODS_CFG[c]['p']
                inv[c] = 0.0

        # 3) 再做基建：由 B/K 配比短板决定“完成的基建吨位”
        # 可用材料：只动用超出安全库存 SS 的部分（保守，便于论证）
        if annual_build_target > 0:
            avail_B = max(0.0, inv['B'] - SS['B'])
            avail_K = max(0.0, inv['K'] - SS['K'])

            need_B = annual_build_target * RHO_BUILD['B']
            need_K = annual_build_target * RHO_BUILD['K']

            use_B = min(need_B, avail_B)
            use_K = min(need_K, avail_K)

            # 完成基建吨位（短板）
            # build_done * rho = use => build_done = use/rho
            build_done_B = use_B / max(RHO_BUILD['B'], 1e-9)
            if RHO_BUILD['K'] > 0:
                build_done_K = use_K / max(RHO_BUILD['K'], 1e-9)
                build_done = min(build_done_B, build_done_K)
            else:
                build_done = build_done_B

            build_done = max(0.0, min(build_done, annual_build_target))

            # 按完成量回扣材料，保持配比一致（防止“扣多了”）
            inv['B'] -= build_done * RHO_BUILD['B']
            inv['K'] -= build_done * RHO_BUILD['K']
            cum_M += build_done

        cum_stockout += stockout_cost

        # --- C. TR 分配 ---
        TR_total_cap = sum(s.get_avail_cap(t, cfg.TR_age_fn, cfg.TR_fail_fn) for s in SITE_NETWORK)
        TR_budget_Moon = TR_total_cap * alpha
        TR_budget_Apex = TR_total_cap * (1 - alpha)

        requests: List[Dict] = []

        # Gap requests
        for c in ['U', 'K', 'B']:
            if gaps[c] > 0:
                is_urgent = (gaps[c] > 0.5 * SS[c]) or (c == 'U')
                mode = 'D' if is_urgent else 'B'
                if c != 'U' and 'B' not in ROUTING[c]:
                    mode = 'D'
                priority = -1e9 if c == 'U' else -gaps[c]
                requests.append({'type': 'gap', 'c': c, 'amt': gaps[c], 'mode': mode, 'prio': priority})

        # Push requests（绑定 remain_goal，防止收尾堆库）
        remain_goal = max(0.0, p_M_total - cum_M)
        if remain_goal > 0:
            push_amt = min(build_cap_limit, remain_goal, TR_budget_Moon * 0.6)
            if push_amt > 0:
                requests.append({'type': 'push', 'c': 'B', 'amt': push_amt, 'mode': 'B', 'prio': 1e9})

        requests.sort(key=lambda x: x['prio'])

        tr_moon_shipped = 0.0
        tr_moon_cost = 0.0
        tr_moon_carb = 0.0
        rem_budget_moon = TR_budget_Moon

        for req in requests:
            c, mode, amt_req = req['c'], req['mode'], req['amt']
            if rem_budget_moon <= 0:
                break

            amt_to_ship = amt_req
            while amt_to_ship > 0 and rem_budget_moon > 0:
                candidates = []
                for s in SITE_NETWORK:
                    cap = s.get_avail_cap(t, cfg.TR_age_fn, cfg.TR_fail_fn)
                    if cap > 0:
                        cost = s.get_effective_cost(p_cTR, MODES[mode]['CostF'])
                        carb = s.get_effective_carb(p_eTR, MODES[mode]['CarbF'])
                        score = cost + W_TIME * MODES[mode]['L'] + W_CARB * carb
                        candidates.append((score, s, cap))

                if not candidates:
                    break

                candidates.sort(key=lambda x: x[0])
                _, s_best, cap_best = candidates[0]

                amount = min(cap_best, amt_to_ship, rem_budget_moon)

                s_best.current_used += amount
                rem_budget_moon -= amount
                amt_to_ship -= amount
                if req['type'] == 'gap':
                    gaps[c] -= amount

                tr_moon_shipped += amount
                tr_moon_cost += amount * s_best.get_effective_cost(p_cTR, MODES[mode]['CostF'])
                tr_moon_carb += amount * s_best.get_effective_carb(p_eTR, MODES[mode]['CarbF'])

                for yr, val in split_arrival(amount, MODES[mode]['L']):
                    pipeline[c][t + yr] = pipeline[c].get(t + yr, 0.0) + val

        # TR -> Apex (投资)
        tr_apex_shipped = 0.0
        tr_apex_cost = 0.0
        tr_apex_carb = 0.0
        rem_budget_apex = TR_budget_Apex
        for s in SITE_NETWORK:
            if rem_budget_apex <= 0:
                break
            cap = s.get_avail_cap(t, cfg.TR_age_fn, cfg.TR_fail_fn)
            if cap > 0:
                amount = min(cap, rem_budget_apex)
                s.current_used += amount
                rem_budget_apex -= amount

                tr_apex_shipped += amount
                tr_apex_cost += amount * s.get_effective_cost(p_cTR, cTR_Apex_Ratio)
                tr_apex_carb += amount * s.get_effective_carb(p_eTR, cTR_Apex_Ratio)

        # --- D. SE 分配 ---
        active_ports = cfg.SE_ports_count_fn(t)
        SE_theory = CSE_THEORY * (active_ports / PORTS_DEFAULT)
        SE_avail = SE_theory * state['A'] * state['eta'] * state['s']

        SE_budget_Moon = SE_avail * beta
        SE_budget_Invest = SE_avail * (1 - beta)

        se_moon_shipped = 0.0
        rem_se = SE_budget_Moon

        for c in ['K', 'B']:
            if gaps[c] > 0 and rem_se > 0:
                amt = min(gaps[c], rem_se)
                rem_se -= amt
                gaps[c] -= amt
                se_moon_shipped += amt
                for yr, val in split_arrival(amt, MODES['SE']['L']):
                    pipeline[c][t + yr] = pipeline[c].get(t + yr, 0.0) + val

        # SE push（同样绑定 remain_goal & 施工能力）
        remain_goal = max(0.0, p_M_total - cum_M)
        if rem_se > 0 and remain_goal > 0:
            push = min(rem_se, remain_goal, build_cap_limit * 0.5)
            if push > 0:
                se_moon_shipped += push
                for yr, val in split_arrival(push, MODES['SE']['L']):
                    pipeline['B'][t + yr] = pipeline['B'].get(t + yr, 0.0) + val

        se_invest_shipped = SE_budget_Invest

        # --- E. 状态演化 (投资效率) ---
        eff_invest = se_invest_shipped * p_inv_eff
        I_total = tr_apex_shipped + eff_invest
        if not cfg.is_perfect_Q1:
            state = update_se_state(state, I_total)

        # --- F. 财务 ---
        se_load = se_moon_shipped + se_invest_shipped
        cost_se = se_load * cSE_op
        carb_se = se_load * eSE_op

        ferry_load = se_moon_shipped
        ferry_fuel = ferry_load * FUEL_RATIO
        cost_ferry = ferry_fuel * p_fuel
        carb_ferry = ferry_fuel * 400.0

        hold_cost = sum(inv[c] for c in GOODS_CFG) * H_HOLD
        carb_tot = tr_moon_carb + tr_apex_carb + carb_se + carb_ferry
        tax = (carb_tot / 1000.0) * p_tax

        cost_tot = tr_moon_cost + tr_apex_cost + cost_se + cost_ferry + hold_cost + stockout_cost + tax
        npv = cost_tot / ((1 + p_disc) ** (t - 2050))
        cum_NPV += npv
        cum_C += carb_tot

        history.append({
            'Year': t,
            'Cum_M': cum_M,
            'Cost_Year': cost_tot,
            'NPV': cum_NPV,
            'Alpha': alpha,
            'Beta': beta,
            'Stockout': stockout_cost,
            'SE_Moon': se_moon_shipped,
            'TR_Moon': tr_moon_shipped,
            'Invest_Ton': I_total,
            'SE_State_s': state['s'],
            'm_pc': curr_m_pc,
            'build_cap': build_cap_limit,
            'remain_goal': max(0.0, p_M_total - cum_M),
        })

        if cum_M >= p_M_total:
            finish_year = t
            is_finished = True
            break

    return {
        'Duration': finish_year - 2050,
        'Total_NPV': cum_NPV,
        'Total_Carbon': cum_C,
        'Total_Stockout_Cost': cum_stockout,
        'Finished': is_finished,
        'Final_SE_S': state['s']
    }, history

# ==============================================================================
# 5. 全能分析工具箱 (Scanner + Sensitivity + Robustness)
# ==============================================================================
def run_parameter_scan(cfg):
    print("\n>>> 1. Running Parameter Scan...")
    alphas = [round(x * 0.05, 2) for x in range(21)]
    betas = [0.8, 0.85, 0.9, 0.95, 1.0]
    results = []

    for a in alphas:
        for b in betas:
            s, _ = simulate_scenario(a, b, cfg)
            s['Alpha'], s['Beta'] = a, b
            results.append(s)

    keys = results[0].keys()
    with open('Q1_Scan.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(results)
    print("Saved Q1_Scan.csv")

def run_sensitivity(base_a, base_b, cfg):
    print("\n>>> 2. Running Sensitivity Analysis...")
    scenarios = [
        ('Base', {}),
        ('Tax_High', {'CARBON_TAX': 100}),
        ('Fuel_High', {'FUEL_PRICE': 1.6e5}),
        ('Cost_TR_Low', {'cTR_Base': 1.0e6}),
        ('Eff_Low', {'SE_INV_EFF': 0.5}),
    ]
    results = []
    for name, ovr in scenarios:
        s, _ = simulate_scenario(base_a, base_b, cfg, overrides=ovr)
        results.append({'Scenario': name, 'NPV': s['Total_NPV'], 'Duration': s['Duration'], 'Finished': s['Finished']})

    keys = results[0].keys()
    with open('Q1_Sensitivity.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(results)
    print("Saved Q1_Sensitivity.csv")

def run_robustness(base_a, base_b, cfg, n_trials=50):
    print(f"\n>>> 3. Running Robustness Check (N={n_trials})...")
    results = []
    for i in range(n_trials):
        r_fail = random.uniform(0.9, 1.0)
        r_se = random.uniform(0.8, 1.1)
        noisy_cfg = SimConfig(
            is_perfect_Q1=False,
            TR_age_fn=cfg.TR_age_fn,
            TR_fail_fn=lambda t: 0.95 * r_fail,
            SE_ports_count_fn=lambda t: max(0, min(PORTS_DEFAULT, int(round(PORTS_DEFAULT * r_se))))
        )
        s, _ = simulate_scenario(base_a, base_b, noisy_cfg)
        results.append({'Trial': i, 'Duration': s['Duration'], 'NPV': s['Total_NPV'], 'Finished': s['Finished']})

    keys = results[0].keys()
    with open('Q1_Robustness.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(results)
    print("Saved Q1_Robustness.csv")

# ==============================================================================
# 6. 主程序
# ==============================================================================
def main():
    print("Start V9.8.1 Final Patch Simulation...")
    cfg = get_config_Q2()

    # 1) 扫描寻找最优解
    run_parameter_scan(cfg)

    # 2) 导出“候选最优策略”轨迹（你可按 scan 结果改）
    best_a, best_b = 0.45, 0.9
    print("\nExporting Best Trajectory...")
    s_best, h_best = simulate_scenario(best_a, best_b, cfg)
    with open('Q1_Best_Trajectory.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, h_best[0].keys())
        w.writeheader()
        w.writerows(h_best)

    # 3) 敏感性 + 鲁棒性
    run_sensitivity(best_a, best_b, cfg)
    run_robustness(best_a, best_b, cfg)

    print("\nAll Done. Good luck with the paper!")
    print(f"[BestCandidate] A={best_a}, B={best_b} -> Duration={s_best['Duration']}, NPV={s_best['Total_NPV']:.3e}, Finished={s_best['Finished']}")

if __name__ == "__main__":
    main()
