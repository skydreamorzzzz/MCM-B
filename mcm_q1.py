# -*- coding: utf-8 -*-
"""
MCM 2026 Problem B — Paper-Friendly Planner (V10.1 Realistic)

Goal: finish M_TOTAL_GOAL tons within a target horizon (Q1 default ~150y).

Design philosophy (paper-friendly, explainable, NOT overly complex):
1) No pipeline delay: delivered this year is usable this year. (Upper-bound-ish, but still realistic by capacities.)
2) No safety stock constraint in construction: all delivered B/K can go to build (after optional sustain reservation).
3) Sustain module default OFF (can enable for sensitivity, still simple).
4) Routing ignores carbon in Q1 score (as requirement). Carbon tax remains in cost.

Key "good points" kept:
- Stoichiometric construction (B/K limiting factor): build is limited by shortboard.
- Push bounded by remaining goal (avoid end-game overstock artifacts).
- Pull driven by annual build target (logistic construction capacity).
- SE_INV_EFF investment effectiveness (kept as a simple multiplier on SE invest effect).
- Parameter scan + sensitivity overrides + robustness Monte Carlo
- Rich history for explainable plots/tables

Outputs:
- Q1_Scan_Final.csv
- Q1_Best_Trajectory.csv
- Q1_Sensitivity.csv
- Q1_Robustness.csv
"""

import math
import csv
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ==============================================================================
# 0) Global Parameters (Realistic Defaults)
# ==============================================================================
# Mission goal
M_TOTAL_GOAL = 1e8          # total infrastructure tonnage
T_MAX = 250                 # simulation horizon
START_YEAR = 2050

# -------- Construction capacity (THIS WAS THE MAIN "too fast" knob before) ----
# Choose mature upper bound ~ 1.0-2.0 Mt/year to make 80-150y plausible for 1e8.
MAX_ANNUAL_BUILD = 1.2e6    # realistic-ish mature build ceiling (ton/year)
LOGISTIC_BUILD_K = 0.15
LOGISTIC_BUILD_MID = 2076   # later maturity than "fastpaper"

# Supply chain ratios for construction (stoichiometric limiting)
RHO_BUILD = {'U': 0.00, 'K': 0.10, 'B': 0.90}  # build 1 ton needs 0.9B + 0.1K

# -------- Optional sustain module (still simple; default OFF) -----------------
ENABLE_SUSTAIN = False
P0 = 1e5
LOGISTIC_DEMAND_K = 0.2
LOGISTIC_DEMAND_MID = 2065
RHO_SUS = {'U': 0.02, 'K': 0.18, 'B': 0.80}
GOODS_PENALTY = {'U': 2e6, 'K': 5e5, 'B': 1e5}

# -------- Modes (kept for cost/carb factors; planner uses bulk "B") -----------
MODES = {
    'D':  {'L': 0.25, 'CostF': 1.00, 'CarbF': 1.00},
    'B':  {'L': 0.50, 'CostF': 0.85, 'CarbF': 0.70},
    'SE': {'L': 0.10, 'CostF': 'Calc', 'CarbF': 'Calc'}
}

# Routing score weights (Q1 ignores carbon by requirement)
W_TIME = 1e5
W_CARB = 5.0

# -------- Traditional rocket (TR) economics (paper: "effective marginal") -----
# Keep costs in a plausible order, and do sensitivity later.
cTR_Base = 3.0e6            # $/ton (baseline TR moon cargo)
cTR_Apex_Ratio = 0.65       # apex leg relative cost
eTR_Base = 5000.0           # kgCO2e/ton (proxy)

# Discount & carbon tax
R_DISC = 0.05
CARBON_TAX = 50.0

# Latitude penalty (small)
LAT_PENALTY_WEIGHT = 0.03

# TR capacity growth (was 0.06 too strong); use 0.02/yr linear multiplier
TR_TECH_GROWTH = 0.02

# -------- Space elevator baseline anchored to the problem statement ----------
# Problem statement: "Galactic Harbour system can transport 179,000 tons/year".
# We anchor total SE system throughput to 1.79e5 ton/year by default.
SE_TOTAL_CAP_ANCHOR = 1.79e5

# We keep "ports" notion for paper (3 harbours), split anchor across ports.
PORTS_DEFAULT = 3
SE_CAP_PER_PORT_BASE = SE_TOTAL_CAP_ANCHOR / PORTS_DEFAULT  # ≈ 59666.7 ton/yr/port

# SE OPEX & parasitic payload (kept simple but physical-ish)
Re, h_apex = 6.37e6, 1e8
r_apex = Re + h_apex
mu_e, omega, g0 = 3.986e14, 7.292e-5, 9.81

# Ferry from apex to lunar transfer: simplified dV & Isp for fuel ratio
dV_ferry = 1800.0
Isp_ferry = 350.0
FUEL_PRICE = 8e4            # $/ton fuel
FUEL_RATIO = math.exp(dV_ferry / (Isp_ferry * g0)) - 1

# SE OPEX (energy proxy)
W_JKG_SAFE_MAX = 1.8e8
E_PRICE = 0.12
ENERGY_EFF = 0.9
C_SE_OPEX_BASE = 8e3
eSE_base = 20.0
eSE_energy = 0.05

# SE investment effectiveness (kept)
SE_INV_EFF = 0.8

# Q1 cap multiplier (keep conservative = 1.0)
CAP_MULT_Q1 = 1.0


# ==============================================================================
# 1) Launch Site Network (exactly 10)
#    Realistic scaling: previous "hundreds of thousands ton/year/site" is too high.
#    We scale down to tens of thousands ton/year/site.
# ==============================================================================
RAW_SITES = [
    ("Alaska_US",           64.8, 260000),
    ("California_US",       34.7, 320000),
    ("Texas_US",            26.0, 700000),
    ("Florida_US",          28.5, 700000),
    ("Virginia_US",         37.0, 320000),
    ("Baikonur_Kazakhstan", 46.0, 450000),
    ("French_Guiana",        5.2, 520000),
    ("Satish_Dhawan_India", 13.7, 420000),
    ("Taiyuan_China",       38.8, 420000),
    ("Mahia_NewZealand",   -39.0, 320000),
]

# Capacity scale factor for realism
TR_CAP_SCALE = 0.08  # 8% of previous: brings global TR to ~0.2-0.6 Mt/year range


@dataclass
class TRSite:
    name: str
    lat: float
    base_cap: float
    k_lat: float = field(init=False)

    def __post_init__(self):
        cos_val = math.cos(math.radians(self.lat))
        self.k_lat = 1.0 - LAT_PENALTY_WEIGHT * (1.0 - cos_val)
        self.k_lat = min(1.0, max(self.k_lat, 0.92))

    def eff_cost(self, base_cost: float, mode_factor: float) -> float:
        return (base_cost * mode_factor) / max(self.k_lat, 1e-9)

    def eff_carb(self, base_carb: float, mode_factor: float) -> float:
        return (base_carb * mode_factor) / max(self.k_lat, 1e-9)


SITE_NETWORK: List[TRSite] = [
    TRSite(n, l, c * TR_CAP_SCALE) for n, l, c in RAW_SITES
]
# sort by effective cost (paper: "best sites utilized first")
SITE_NETWORK.sort(key=lambda s: s.eff_cost(cTR_Base, 1.0))


# ==============================================================================
# 2) Helpers
# ==============================================================================
def logistic_curve(t: int, start_val: float, end_val: float, midpoint: int, k: float) -> float:
    """A smooth ramp from start_val to end_val."""
    return end_val + (start_val - end_val) / (1 + math.exp(k * (t - midpoint)))


def unit_work_J_per_kg() -> float:
    """Energy to raise 1 kg from Earth to apex (proxy)."""
    return mu_e * (1 / Re - 1 / r_apex) - 0.5 * omega**2 * (r_apex**2 - Re**2)


def calc_se_opex(ovr: Optional[Dict] = None):
    """SE cost/carbon per ton lifted, with safe clamp."""
    price = (ovr.get('E_PRICE', E_PRICE) if ovr else E_PRICE)
    eff = (ovr.get('ENERGY_EFF', ENERGY_EFF) if ovr else ENERGY_EFF)
    base = (ovr.get('C_SE_OPEX_BASE', C_SE_OPEX_BASE) if ovr else C_SE_OPEX_BASE)

    W = max(0.0, min(unit_work_J_per_kg(), W_JKG_SAFE_MAX))
    kWh_ton = (W / 3.6e6) * 1000.0
    cost = (kWh_ton * price) / max(eff, 1e-9) + base
    carb = eSE_base + kWh_ton * eSE_energy
    return cost, carb


def cheapest_tr_unit(base_cost: float, mode: str) -> float:
    best = SITE_NETWORK[0]
    return best.eff_cost(base_cost, MODES[mode]['CostF'])


def cheapest_tr_carb(base_carb: float, mode: str) -> float:
    best = SITE_NETWORK[0]
    return best.eff_carb(base_carb, MODES[mode]['CarbF'])


# ==============================================================================
# 3) Config
# ==============================================================================
@dataclass
class SimConfig:
    is_Q1: bool = True
    consider_carbon_in_routing_score: bool = False
    enable_sustain: bool = ENABLE_SUSTAIN
    cap_mult_q1: float = CAP_MULT_Q1


def get_config_Q1():
    return SimConfig(is_Q1=True,
                     consider_carbon_in_routing_score=False,
                     enable_sustain=ENABLE_SUSTAIN,
                     cap_mult_q1=CAP_MULT_Q1)


def get_config_Q2():
    return SimConfig(is_Q1=False,
                     consider_carbon_in_routing_score=False,
                     enable_sustain=ENABLE_SUSTAIN,
                     cap_mult_q1=CAP_MULT_Q1)


# ==============================================================================
# 4) Simulation Engine (paper-friendly, no pipeline delay)
# ==============================================================================
def simulate_scenario(alpha: float, beta: float, cfg: SimConfig, overrides: Optional[Dict] = None):
    """
    Decision variables:
    alpha: fraction of TR capacity allocated to Moon cargo (rest to apex support)
    beta:  fraction of SE capacity used for Moon lift (rest for investment/support lift)

    State:
    cum_M: completed infrastructure tonnage
    """
    ovr = overrides if overrides else {}

    # overrides
    p_cTR = ovr.get('cTR_Base', cTR_Base)
    p_eTR = ovr.get('eTR_Base', eTR_Base)
    p_tax = ovr.get('CARBON_TAX', CARBON_TAX)
    p_disc = ovr.get('R_DISC', R_DISC)
    p_M_total = ovr.get('M_TOTAL_GOAL', M_TOTAL_GOAL)
    p_fuel = ovr.get('FUEL_PRICE', FUEL_PRICE)

    # construction
    p_max_build = ovr.get('MAX_ANNUAL_BUILD', MAX_ANNUAL_BUILD)
    p_log_mid = ovr.get('LOGISTIC_BUILD_MID', LOGISTIC_BUILD_MID)
    p_log_k = ovr.get('LOGISTIC_BUILD_K', LOGISTIC_BUILD_K)

    # elevator capacity anchored
    ports = int(ovr.get('PORTS_DEFAULT', PORTS_DEFAULT))
    se_cap_per_port = float(ovr.get('SE_CAP_PER_PORT_BASE', SE_CAP_PER_PORT_BASE))
    cse_base = ports * se_cap_per_port

    cSE_op, eSE_op = calc_se_opex(ovr)

    # Q1: fixed cap_mult
    if cfg.is_Q1:
        cap_mult = max(1.0, float(cfg.cap_mult_q1))
        # investment effectiveness: treat as a simple multiplier on invest budget
        A, eta = 1.0, 1.0
    else:
        cap_mult = 1.0
        A, eta = 0.9, 0.8

    cum_M = 0.0
    cum_NPV = 0.0
    cum_Carb = 0.0
    cum_undisc = 0.0

    history = []
    finish_year = START_YEAR + T_MAX
    is_finished = False

    for t in range(START_YEAR, START_YEAR + T_MAX + 1):
        # --- Annual build target via logistic construction capacity ---
        build_cap_limit = logistic_curve(
            t,
            p_max_build * 0.15,   # early stage smaller
            p_max_build,
            p_log_mid,
            p_log_k
        )
        remain_goal = max(0.0, p_M_total - cum_M)
        annual_build_target = min(build_cap_limit, remain_goal)

        # --- Optional sustain (simple reservation) ---
        stockout_cost = 0.0
        if cfg.enable_sustain:
            curr_m_pc = logistic_curve(t, 2.0, 0.5, LOGISTIC_DEMAND_MID, LOGISTIC_DEMAND_K)
            D_sus_tot = P0 * curr_m_pc
            sus_B = D_sus_tot * RHO_SUS['B']
            sus_K = D_sus_tot * RHO_SUS['K']
        else:
            D_sus_tot = 0.0
            sus_B = 0.0
            sus_K = 0.0

        # --- Construction goods requirement ---
        need_B = annual_build_target * RHO_BUILD['B']
        need_K = annual_build_target * RHO_BUILD['K']

        # --- TR total capacity with mild tech progress ---
        mult = 1.0 + TR_TECH_GROWTH * (t - START_YEAR)
        TR_total_cap = sum(s.base_cap * s.k_lat for s in SITE_NETWORK) * mult
        TR_budget_moon = TR_total_cap * alpha
        TR_budget_apex = TR_total_cap * (1.0 - alpha)

        # --- SE total capacity (anchored) ---
        SE_total = cse_base * cap_mult * A * eta
        SE_budget_invest = SE_total * (1.0 - beta)
        SE_budget_moon_total_lift = SE_total * beta
        SE_budget_moon_cargo = SE_budget_moon_total_lift / (1.0 + FUEL_RATIO)

        # ---- Allocation Policy (paper-friendly planner) ----
        # Priority: sustain first, then construction K then B.
        # Any surplus cargo: push B but bounded by remain_goal.

        # 1) SE cargo -> K then B
        K_from_SE = min(SE_budget_moon_cargo, sus_K + need_K)
        rem_SE = SE_budget_moon_cargo - K_from_SE

        B_from_SE = min(rem_SE, sus_B + need_B)
        rem_SE -= B_from_SE

        push_B_SE = min(rem_SE, remain_goal * RHO_BUILD['B'])
        B_from_SE += push_B_SE

        # 2) TR moon cargo -> K then B
        K_from_TR = min(TR_budget_moon, sus_K + max(0.0, need_K - K_from_SE))
        rem_TR = TR_budget_moon - K_from_TR

        B_from_TR = min(rem_TR, sus_B + max(0.0, need_B - B_from_SE))
        rem_TR -= B_from_TR

        push_B_TR = min(rem_TR, remain_goal * RHO_BUILD['B'])
        B_from_TR += push_B_TR

        # Delivered to Moon this year (usable immediately)
        delivered_K = max(0.0, K_from_SE + K_from_TR)
        delivered_B = max(0.0, B_from_SE + B_from_TR)

        # sustain feasibility penalty
        if cfg.enable_sustain:
            if delivered_K < sus_K:
                stockout_cost += (sus_K - delivered_K) * GOODS_PENALTY['K']
            if delivered_B < sus_B:
                stockout_cost += (sus_B - delivered_B) * GOODS_PENALTY['B']

        # Available for construction after sustain reservation
        avail_K = max(0.0, delivered_K - sus_K)
        avail_B = max(0.0, delivered_B - sus_B)

        # Stoichiometric construction (shortboard)
        build_done = 0.0
        if annual_build_target > 0:
            max_by_B = avail_B / max(RHO_BUILD['B'], 1e-12)
            max_by_K = avail_K / max(RHO_BUILD['K'], 1e-12)
            build_done = min(annual_build_target, max_by_B, max_by_K)
            build_done = max(0.0, build_done)
            cum_M += build_done

        # ---- Costs (simple + explainable) ----
        unit_tr_cost = cheapest_tr_unit(p_cTR, 'B')
        unit_tr_carb = cheapest_tr_carb(p_eTR, 'B')
        tr_moon_ship = (K_from_TR + B_from_TR)

        tr_moon_cost = tr_moon_ship * unit_tr_cost
        tr_moon_carb = tr_moon_ship * unit_tr_carb

        # TR apex (support) cost proxy
        tr_apex_cost = TR_budget_apex * cheapest_tr_unit(p_cTR * cTR_Apex_Ratio, 'B')
        tr_apex_carb = TR_budget_apex * cheapest_tr_carb(p_eTR * cTR_Apex_Ratio, 'B')

        # SE opex: total lift includes parasitic ferry fuel mass
        se_moon_total_lift = (K_from_SE + B_from_SE) * (1.0 + FUEL_RATIO)
        # investment lift scaled by effectiveness (paper: "investment tonnage produces SE_INV_EFF effect")
        se_invest_lift = SE_budget_invest
        se_load_total = se_moon_total_lift + se_invest_lift

        cost_se = se_load_total * cSE_op
        carb_se = se_load_total * eSE_op

        # ferry fuel purchase
        ferry_fuel = (K_from_SE + B_from_SE) * FUEL_RATIO
        cost_ferry = ferry_fuel * p_fuel
        carb_ferry = ferry_fuel * 400.0  # proxy

        carb_tot = tr_moon_carb + tr_apex_carb + carb_se + carb_ferry
        tax = (carb_tot / 1000.0) * p_tax

        cost_tot = tr_moon_cost + tr_apex_cost + cost_se + cost_ferry + stockout_cost + tax

        npv_add = cost_tot / ((1.0 + p_disc) ** (t - START_YEAR))
        cum_NPV += npv_add
        cum_undisc += cost_tot
        cum_Carb += carb_tot

        history.append({
            'Year': t,
            'Alpha': alpha, 'Beta': beta,
            'TR_total_cap': TR_total_cap,
            'TR_moon': tr_moon_ship,
            'TR_apex': TR_budget_apex,
            'SE_total': SE_total,
            'SE_moon_cargo': (K_from_SE + B_from_SE),
            'SE_invest': SE_budget_invest,
            'Delivered_K': delivered_K,
            'Delivered_B': delivered_B,
            'Build_Target': annual_build_target,
            'Build_Done': build_done,
            'Cum_M': cum_M,
            'Remain_Goal': max(0.0, p_M_total - cum_M),
            'Cost_Year': cost_tot,
            'NPV_Cum': cum_NPV,
            'Carb_Cum': cum_Carb
        })

        if cum_M >= p_M_total - 1e-9:
            finish_year = t
            is_finished = True
            break

    summary = {
        'Duration': finish_year - START_YEAR,
        'Total_NPV': cum_NPV,
        'Total_Carbon': cum_Carb,
        'Total_Undiscounted_Cost': cum_undisc,
        'Finished': is_finished,
        'Alpha': alpha,
        'Beta': beta
    }
    return summary, history


# ==============================================================================
# 5) Analysis Tools
# ==============================================================================
def run_parameter_scan(cfg: SimConfig, out_csv='Q1_Scan_Final.csv'):
    print(f">>> Running Parameter Scan -> {out_csv}")
    alphas = [round(x * 0.05, 2) for x in range(21)]  # 0..1 step 0.05
    betas = [0.8, 0.85, 0.9, 0.95, 1.0]
    results = []
    for a in alphas:
        for b in betas:
            s, _ = simulate_scenario(a, b, cfg)
            results.append({
                'Duration': s['Duration'],
                'Total_NPV': s['Total_NPV'],
                'Total_Carbon': s['Total_Carbon'],
                'Total_Undiscounted_Cost': s['Total_Undiscounted_Cost'],
                'Finished': s['Finished'],
                'Alpha': a, 'Beta': b
            })

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("Saved:", out_csv)


def run_sensitivity(base_a, base_b, cfg: SimConfig, out_csv='Q1_Sensitivity.csv'):
    print(f">>> Running Sensitivity -> {out_csv}")
    scenarios = [
        ('Base', {}),
        ('Tax_High', {'CARBON_TAX': 120}),
        ('Fuel_High', {'FUEL_PRICE': 1.6e5}),
        ('TR_Cost_Low', {'cTR_Base': 2.0e6}),
        ('TR_Cost_High', {'cTR_Base': 6.0e6}),
        ('TR_Growth_Low', {'TR_TECH_GROWTH': 0.01}),  # handled below manually
        ('TR_Growth_High', {'TR_TECH_GROWTH': 0.03}),
        ('SE_Cap_Low', {'SE_CAP_PER_PORT_BASE': (SE_TOTAL_CAP_ANCHOR * 0.7) / PORTS_DEFAULT}),
        ('SE_Cap_High', {'SE_CAP_PER_PORT_BASE': (SE_TOTAL_CAP_ANCHOR * 1.3) / PORTS_DEFAULT}),
        ('BuildCap_Low', {'MAX_ANNUAL_BUILD': 8e5}),
        ('BuildCap_High', {'MAX_ANNUAL_BUILD': 1.8e6}),
    ]

    rows = []
    global TR_TECH_GROWTH
    saved_growth = TR_TECH_GROWTH

    for name, ovr in scenarios:
        # Special-case TR growth override (global in this file)
        if 'TR_TECH_GROWTH' in ovr:
            TR_TECH_GROWTH = float(ovr['TR_TECH_GROWTH'])
            ovr2 = dict(ovr)
            del ovr2['TR_TECH_GROWTH']
        else:
            ovr2 = ovr

        s, _ = simulate_scenario(base_a, base_b, cfg, overrides=ovr2)
        rows.append({
            'Scenario': name,
            'Alpha': base_a, 'Beta': base_b,
            'Duration': s['Duration'],
            'Finished': s['Finished'],
            'Total_NPV': s['Total_NPV'],
            'Total_Undiscounted_Cost': s['Total_Undiscounted_Cost'],
            'Total_Carbon': s['Total_Carbon'],
        })

    TR_TECH_GROWTH = saved_growth

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out_csv)


def run_robustness(base_a, base_b, cfg: SimConfig, n_trials=50, out_csv='Q1_Robustness.csv'):
    print(f">>> Running Robustness N={n_trials} -> {out_csv}")
    rows = []
    for i in range(n_trials):
        # Randomize: SE total capacity around anchor ±15%, build cap ±15%
        r_se = random.uniform(0.85, 1.15)
        r_build = random.uniform(0.85, 1.15)

        ovr = {
            'SE_CAP_PER_PORT_BASE': (SE_TOTAL_CAP_ANCHOR * r_se) / PORTS_DEFAULT,
            'MAX_ANNUAL_BUILD': MAX_ANNUAL_BUILD * r_build
        }

        s, _ = simulate_scenario(base_a, base_b, cfg, overrides=ovr)
        rows.append({
            'Trial': i,
            'r_se': r_se,
            'r_build': r_build,
            'Duration': s['Duration'],
            'Finished': s['Finished'],
            'Total_NPV': s['Total_NPV'],
            'Total_Carbon': s['Total_Carbon'],
        })

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out_csv)


# ==============================================================================
# 6) Main
# ==============================================================================
def main():
    print("Start V10.1 Realistic Paper Planner...")
    cfg = get_config_Q1()

    # scan
    run_parameter_scan(cfg, out_csv='Q1_Scan_Final.csv')

    # pick a candidate (you can change after seeing scan results)
    # heuristic: give more TR to Moon, and SE mostly to Moon
    best_a, best_b = 0.55, 0.95
    s_best, h_best = simulate_scenario(best_a, best_b, cfg)

    with open('Q1_Best_Trajectory.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(h_best[0].keys()))
        w.writeheader()
        w.writerows(h_best)

    run_sensitivity(best_a, best_b, cfg, out_csv='Q1_Sensitivity.csv')
    run_robustness(best_a, best_b, cfg, n_trials=50, out_csv='Q1_Robustness.csv')

    print("\nAll Done.")
    print(f"[Candidate] A={best_a}, B={best_b} -> "
          f"Duration={s_best['Duration']}, Finished={s_best['Finished']}, "
          f"NPV={s_best['Total_NPV']:.3e}")

if __name__ == "__main__":
    main()
