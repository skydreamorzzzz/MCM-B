# -*- coding: utf-8 -*-
"""
MCM 2026 Problem B — Paper-Friendly Planner (V10.3 Q1-XY-Unified)

What changed vs your V10.2 (minimal + aligned to your modeling lead):
- Replace (alpha, beta) with (x, y) as two "scheme weights / utilization intensities".
  * x in [0,1]: TR (rocket) utilization intensity (0=off, 1=full baseline)
  * y in [0,1]: SE (space elevator) utilization intensity (0=off, 1=full baseline)
- Space elevator DOES NOT grow scale. Maintenance is a constant availability loss:
  * SE_UTIL = 0.95 (constant), NOT a decision variable.
- Remove SE "invest/upgrade" split entirely (no SE_budget_invest, no (1-beta)).
- Keep a simple TR support fraction as a constant (paper-friendly):
  * TR_MOON_FRAC = 0.85 (rest goes to "support/apex logistics" cost proxy)
- Q1 CSV outputs focus on duration + cost + TR/SE contribution shares (no carbon columns).

Outputs:
- Q1_Scan_Final.csv      (scan x,y grid)
- Q1_Best_Trajectory.csv (trajectory for chosen best (x,y))
- Q1_ThreeSchemes.csv    (TR-only / SE-only / Hybrid (x,y) unified)
- Q1_Sensitivity.csv
- Q1_Robustness.csv
"""

import math
import csv
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# 0) Global Parameters (keep your latest edits where possible)
# ==============================================================================
# Mission goal
M_TOTAL_GOAL = 1e8          # total infrastructure tonnage
T_MAX = 1000                # simulation horizon
START_YEAR = 2050

# ---- Unified weights (decision variables) ----
# x: TR utilization intensity in [0,1]
# y: SE utilization intensity in [0,1]

# ---- Constant assumptions for Q1 paper-friendliness ----
SE_UTIL = 0.95              # constant elevator availability after maintenance/overhead
TR_MOON_FRAC = 0.85         # constant fraction of TR capacity used for Moon cargo
# (1-TR_MOON_FRAC) is "support/apex logistics" proxy

# -------- Construction capacity (key realism knob) ----------------------------
MAX_ANNUAL_BUILD = 1.2e6    # mature build ceiling (ton/year)
LOGISTIC_BUILD_K = 0.15
LOGISTIC_BUILD_MID = 2076

# Supply chain ratios for construction (stoichiometric limiting)
RHO_BUILD = {'U': 0.00, 'K': 0.10, 'B': 0.90}  # build 1 ton needs 0.9B + 0.1K

# -------- Optional sustain module (default OFF) -------------------------------
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

# -------- Traditional rocket (TR) economics ----------------------------------
cTR_Base = 3.0e6            # $/ton baseline TR moon cargo
cTR_Apex_Ratio = 0.65       # support/apex leg relative cost
eTR_Base = 5000.0           # kgCO2e/ton (proxy, not output in Q1)

# Discount & carbon tax (carbon not output in Q1 CSVs)
R_DISC = 0.05
CARBON_TAX = 50.0

# Latitude penalty (small)
LAT_PENALTY_WEIGHT = 0.03

# TR capacity growth (mild linear)
TR_TECH_GROWTH = 0.02

# -------- Space elevator baseline (your updated interpretation) ---------------
# You updated system total to 5.37e5 (instead of 1.79e5). Keep it.
SE_TOTAL_CAP_ANCHOR = 5.37e5  # ton/year (SYSTEM total, fixed scale)
PORTS_DEFAULT = 3
SE_CAP_PER_PORT_BASE = SE_TOTAL_CAP_ANCHOR / PORTS_DEFAULT

# SE OPEX & parasitic payload
Re, h_apex = 6.37e6, 1e8
r_apex = Re + h_apex
mu_e, omega, g0 = 3.986e14, 7.292e-5, 9.81

# Ferry from apex to lunar transfer: simplified fuel ratio (your edited Isp)
dV_ferry = 1800.0
Isp_ferry = 2000.0
FUEL_PRICE = 8e4            # $/ton fuel
FUEL_RATIO = math.exp(dV_ferry / (Isp_ferry * g0)) - 1

# SE OPEX (energy proxy)
W_JKG_SAFE_MAX = 1.8e8
E_PRICE = 0.12
ENERGY_EFF = 0.9
C_SE_OPEX_BASE = 8e3
eSE_base = 20.0
eSE_energy = 0.05

# We keep SE_INV_EFF defined but unused in Q1 (no scaling growth)
SE_INV_EFF = 0.8


# ==============================================================================
# 1) Launch Site Network (exactly 10)
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
TR_CAP_SCALE = 0.08


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


SITE_NETWORK: List[TRSite] = [TRSite(n, l, c * TR_CAP_SCALE) for n, l, c in RAW_SITES]
SITE_NETWORK.sort(key=lambda s: s.eff_cost(cTR_Base, 1.0))


# ==============================================================================
# 2) Helpers
# ==============================================================================
def logistic_curve(t: int, start_val: float, end_val: float, midpoint: int, k: float) -> float:
    return end_val + (start_val - end_val) / (1 + math.exp(k * (t - midpoint)))


def unit_work_J_per_kg() -> float:
    return mu_e * (1 / Re - 1 / r_apex) - 0.5 * omega**2 * (r_apex**2 - Re**2)


def calc_se_opex(ovr: Optional[Dict] = None) -> Tuple[float, float]:
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


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


# ==============================================================================
# 3) Config
# ==============================================================================
@dataclass
class SimConfig:
    enable_sustain: bool = ENABLE_SUSTAIN


def get_config_Q1():
    return SimConfig(enable_sustain=ENABLE_SUSTAIN)


# ==============================================================================
# 4) Simulation Engine (paper-friendly, no pipeline delay)
# ==============================================================================
def simulate_scenario(x: float, y: float, cfg: SimConfig, overrides: Optional[Dict] = None):
    """
    x: TR utilization intensity in [0,1]
    y: SE utilization intensity in [0,1]
    """
    ovr = overrides if overrides else {}
    x = clamp01(x)
    y = clamp01(y)

    # overrides: economics
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

    # SE capacity anchored (fixed scale, but allow sensitivity overrides)
    ports = int(ovr.get('PORTS_DEFAULT', PORTS_DEFAULT))
    se_cap_per_port = float(ovr.get('SE_CAP_PER_PORT_BASE', SE_CAP_PER_PORT_BASE))
    se_fixed_total = ports * se_cap_per_port  # fixed baseline (no growth)

    # allow overriding SE utilization constant (rare; keep default)
    se_util = float(ovr.get('SE_UTIL', SE_UTIL))
    se_util = max(0.0, min(1.0, se_util))

    cSE_op, eSE_op = calc_se_opex(ovr)

    cum_M = 0.0
    cum_NPV = 0.0
    cum_undisc = 0.0
    cum_Carb_internal = 0.0  # internal only

    # contribution accounting
    cum_TR_moon_ship = 0.0
    cum_SE_moon_ship = 0.0

    history = []
    finish_year = START_YEAR + T_MAX
    is_finished = False

    for t in range(START_YEAR, START_YEAR + T_MAX + 1):
        # --- Annual build target ---
        build_cap_limit = logistic_curve(
            t,
            p_max_build * 0.15,
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
            sus_B = 0.0
            sus_K = 0.0

        # --- Construction goods requirement ---
        need_B = annual_build_target * RHO_BUILD['B']
        need_K = annual_build_target * RHO_BUILD['K']

        # --- TR total capacity (baseline) with tech progress, then scaled by x ---
        mult = 1.0 + TR_TECH_GROWTH * (t - START_YEAR)
        TR_total_cap_baseline = sum(s.base_cap * s.k_lat for s in SITE_NETWORK) * mult

        # allow overriding TR scale (for stress tests)
        tr_scale_mult = float(ovr.get('TR_SCALE_MULT', 1.0))
        TR_total_cap = TR_total_cap_baseline * max(0.0, tr_scale_mult) * x

        TR_moon_cap = TR_total_cap * TR_MOON_FRAC
        TR_support_cap = TR_total_cap * (1.0 - TR_MOON_FRAC)

        # --- SE moon cargo capacity (fixed scale, no growth), scaled by y and constant util ---
        # total lift available after maintenance: se_fixed_total * se_util * y
        # cargo available after ferry fuel mass: / (1 + FUEL_RATIO)
        SE_moon_cargo_cap = (se_fixed_total * se_util * y) / (1.0 + FUEL_RATIO)

        # ---- Allocation Policy ----
        # Priority: sustain first, then construction K then B.
        # Any surplus cargo: push B but bounded by remain_goal.

        # 1) SE cargo -> K then B
        K_from_SE = min(SE_moon_cargo_cap, sus_K + need_K)
        rem_SE = SE_moon_cargo_cap - K_from_SE

        B_from_SE = min(rem_SE, sus_B + need_B)
        rem_SE -= B_from_SE

        push_B_SE = min(rem_SE, remain_goal * RHO_BUILD['B'])
        B_from_SE += push_B_SE

        # 2) TR moon cargo -> K then B
        K_from_TR = min(TR_moon_cap, sus_K + max(0.0, need_K - K_from_SE))
        rem_TR = TR_moon_cap - K_from_TR

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

        # ---- Costs (computed; carbon not output in Q1 CSVs) ----
        unit_tr_cost = cheapest_tr_unit(p_cTR, 'B')
        unit_tr_carb = cheapest_tr_carb(p_eTR, 'B')

        tr_moon_ship = (K_from_TR + B_from_TR)
        se_moon_ship = (K_from_SE + B_from_SE)

        cum_TR_moon_ship += tr_moon_ship
        cum_SE_moon_ship += se_moon_ship

        # TR Moon cost
        tr_moon_cost = tr_moon_ship * unit_tr_cost
        tr_moon_carb = tr_moon_ship * unit_tr_carb

        # TR support cost proxy (apex/support)
        tr_support_cost = TR_support_cap * cheapest_tr_unit(p_cTR * cTR_Apex_Ratio, 'B')
        tr_support_carb = TR_support_cap * cheapest_tr_carb(p_eTR * cTR_Apex_Ratio, 'B')

        # SE opex: total lift includes parasitic ferry fuel mass
        se_moon_total_lift = se_moon_ship * (1.0 + FUEL_RATIO)
        cost_se = se_moon_total_lift * cSE_op
        carb_se = se_moon_total_lift * eSE_op

        # ferry fuel purchase
        ferry_fuel = se_moon_ship * FUEL_RATIO
        cost_ferry = ferry_fuel * p_fuel
        carb_ferry = ferry_fuel * 400.0  # proxy

        carb_tot = tr_moon_carb + tr_support_carb + carb_se + carb_ferry
        tax = (carb_tot / 1000.0) * p_tax

        cost_tot = tr_moon_cost + tr_support_cost + cost_se + cost_ferry + stockout_cost + tax

        npv_add = cost_tot / ((1.0 + p_disc) ** (t - START_YEAR))
        cum_NPV += npv_add
        cum_undisc += cost_tot
        cum_Carb_internal += carb_tot

        history.append({
            'Year': t,
            'x': x,
            'y': y,
            'TR_total_cap_used': TR_total_cap,
            'TR_moon': tr_moon_ship,
            'TR_support': TR_support_cap,
            'SE_fixed_total': se_fixed_total,
            'SE_moon_cargo': se_moon_ship,
            'Delivered_K': delivered_K,
            'Delivered_B': delivered_B,
            'Build_Target': annual_build_target,
            'Build_Done': build_done,
            'Cum_M': cum_M,
            'Remain_Goal': max(0.0, p_M_total - cum_M),
            'Cost_Year': cost_tot,
            'NPV_Cum': cum_NPV,
            'Carb_Cum_Internal': cum_Carb_internal,
        })

        if cum_M >= p_M_total - 1e-9:
            finish_year = t
            is_finished = True
            break

    total_moon_ship = cum_TR_moon_ship + cum_SE_moon_ship
    tr_share = (cum_TR_moon_ship / total_moon_ship) if total_moon_ship > 0 else 0.0
    se_share = (cum_SE_moon_ship / total_moon_ship) if total_moon_ship > 0 else 0.0

    summary = {
        'Duration': finish_year - START_YEAR,
        'Finished': is_finished,
        'x': x,
        'y': y,
        'Total_NPV': cum_NPV,
        'Total_Undiscounted_Cost': cum_undisc,

        # Contribution (what you want in CSV)
        'TR_Moon_Total': cum_TR_moon_ship,
        'SE_Moon_Total': cum_SE_moon_ship,
        'TR_Share': tr_share,
        'SE_Share': se_share,

        # Internal only (not reported in Q1 CSVs)
        'Total_Carbon_Internal': cum_Carb_internal,
    }
    return summary, history


# ==============================================================================
# 5) Analysis Tools (Q1 reporting: no carbon columns)
# ==============================================================================
def run_parameter_scan(cfg: SimConfig, out_csv='Q1_Scan_Final.csv',
                       step: float = 0.05):
    """
    Scan a grid over (x,y).
    step=0.05 -> 21x21 = 441 cases (still fast).
    """
    print(f">>> Running Q1 Scan over (x,y) grid step={step} -> {out_csv}")
    n = int(round(1.0 / step))
    vals = [round(i * step, 2) for i in range(n + 1)]

    results = []
    for x in vals:
        for y in vals:
            s, _ = simulate_scenario(x, y, cfg)
            results.append({
                'x': x,
                'y': y,
                'Duration': s['Duration'],
                'Finished': s['Finished'],
                'Total_NPV': s['Total_NPV'],
                'Total_Undiscounted_Cost': s['Total_Undiscounted_Cost'],
                'TR_Share': s['TR_Share'],
                'SE_Share': s['SE_Share'],
                'TR_Moon_Total': s['TR_Moon_Total'],
                'SE_Moon_Total': s['SE_Moon_Total'],
            })

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("Saved:", out_csv)
    return results


def pick_best_from_scan(scan_rows: List[Dict]) -> Tuple[float, float, Dict]:
    """
    Paper-friendly rule:
    1) Prefer Finished=True
    2) Minimize Duration
    3) Tie-break by Total_Undiscounted_Cost
    """
    finished = [r for r in scan_rows if r['Finished']]
    pool = finished if finished else scan_rows

    pool.sort(key=lambda r: (r['Duration'], r['Total_Undiscounted_Cost']))
    best = pool[0]
    return float(best['x']), float(best['y']), best


def run_sensitivity(base_x, base_y, cfg: SimConfig, out_csv='Q1_Sensitivity.csv'):
    """
    Sensitivity: duration/cost/shares (no carbon column).
    """
    print(f">>> Running Q1 Sensitivity -> {out_csv}")
    scenarios = [
        ('Base', {}),
        ('Tax_High', {'CARBON_TAX': 120}),
        ('Fuel_High', {'FUEL_PRICE': 1.6e5}),
        ('TR_Cost_Low', {'cTR_Base': 2.0e6}),
        ('TR_Cost_High', {'cTR_Base': 6.0e6}),
        ('SE_Cap_Low', {'SE_CAP_PER_PORT_BASE': (SE_TOTAL_CAP_ANCHOR * 0.7) / PORTS_DEFAULT}),
        ('SE_Cap_High', {'SE_CAP_PER_PORT_BASE': (SE_TOTAL_CAP_ANCHOR * 1.3) / PORTS_DEFAULT}),
        ('BuildCap_Low', {'MAX_ANNUAL_BUILD': 8e5}),
        ('BuildCap_High', {'MAX_ANNUAL_BUILD': 1.8e6}),
        ('TR_MoonFrac_Low', {'TR_MOON_FRAC': 0.75}),  # handled below
        ('TR_MoonFrac_High', {'TR_MOON_FRAC': 0.92}),
    ]

    rows = []
    global TR_MOON_FRAC
    saved_frac = TR_MOON_FRAC

    for name, ovr in scenarios:
        if 'TR_MOON_FRAC' in ovr:
            TR_MOON_FRAC = float(ovr['TR_MOON_FRAC'])
            ovr2 = dict(ovr)
            del ovr2['TR_MOON_FRAC']
        else:
            ovr2 = ovr

        s, _ = simulate_scenario(base_x, base_y, cfg, overrides=ovr2)
        rows.append({
            'Scenario': name,
            'x': base_x,
            'y': base_y,
            'Duration': s['Duration'],
            'Finished': s['Finished'],
            'Total_NPV': s['Total_NPV'],
            'Total_Undiscounted_Cost': s['Total_Undiscounted_Cost'],
            'TR_Share': s['TR_Share'],
            'SE_Share': s['SE_Share'],
        })

    TR_MOON_FRAC = saved_frac

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out_csv)


def run_robustness(base_x, base_y, cfg: SimConfig, n_trials=50, out_csv='Q1_Robustness.csv'):
    """
    Robustness: randomize SE capacity ±15%, build cap ±15%
    """
    print(f">>> Running Q1 Robustness N={n_trials} -> {out_csv}")
    rows = []
    for i in range(n_trials):
        r_se = random.uniform(0.85, 1.15)
        r_build = random.uniform(0.85, 1.15)

        ovr = {
            'SE_CAP_PER_PORT_BASE': (SE_TOTAL_CAP_ANCHOR * r_se) / PORTS_DEFAULT,
            'MAX_ANNUAL_BUILD': MAX_ANNUAL_BUILD * r_build
        }

        s, _ = simulate_scenario(base_x, base_y, cfg, overrides=ovr)
        rows.append({
            'Trial': i,
            'r_se': r_se,
            'r_build': r_build,
            'x': base_x,
            'y': base_y,
            'Duration': s['Duration'],
            'Finished': s['Finished'],
            'Total_NPV': s['Total_NPV'],
            'Total_Undiscounted_Cost': s['Total_Undiscounted_Cost'],
            'TR_Share': s['TR_Share'],
            'SE_Share': s['SE_Share'],
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
    print("Start V10.3 Q1-XY-Unified Paper Planner...")
    cfg = get_config_Q1()

    # 1) Scan (x,y)
    scan_rows = run_parameter_scan(cfg, out_csv='Q1_Scan_Final.csv', step=0.05)

    # 2) Pick best (paper-friendly rule)
    best_x, best_y, best_row = pick_best_from_scan(scan_rows)
    print(f">>> Picked best (x,y)=({best_x},{best_y}) | "
          f"Finished={best_row['Finished']} | Duration={best_row['Duration']}")

    # 3) Best trajectory
    s_best, h_best = simulate_scenario(best_x, best_y, cfg)
    with open('Q1_Best_Trajectory.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(h_best[0].keys()))
        w.writeheader()
        w.writerows(h_best)
    print("Saved: Q1_Best_Trajectory.csv")

    # 4) Three schemes unified in (x,y)
    # TR-only: (1,0); SE-only: (0,1); Hybrid: (1,1)
    schemes = [
        ('TR_only', 1.0, 0.0),
        ('SE_only', 0.0, 1.0),
        ('Hybrid',  1.0, 1.0),
    ]
    three_rows = []
    for name, x, y in schemes:
        s, _ = simulate_scenario(x, y, cfg)
        three_rows.append({
            'Scheme': name,
            'x': x,
            'y': y,
            'Duration': s['Duration'],
            'Finished': s['Finished'],
            'TR_Share': s['TR_Share'],
            'SE_Share': s['SE_Share'],
            'TR_Moon_Total': s['TR_Moon_Total'],
            'SE_Moon_Total': s['SE_Moon_Total'],
            'Total_NPV': s['Total_NPV'],
            'Total_Undiscounted_Cost': s['Total_Undiscounted_Cost'],
        })

    with open('Q1_ThreeSchemes.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(three_rows[0].keys()))
        w.writeheader()
        w.writerows(three_rows)
    print("Saved: Q1_ThreeSchemes.csv")

    # 5) Sensitivity / Robustness around best (x,y)
    run_sensitivity(best_x, best_y, cfg, out_csv='Q1_Sensitivity.csv')
    run_robustness(best_x, best_y, cfg, n_trials=50, out_csv='Q1_Robustness.csv')

    print("\nAll Done.")
    print(f"[Best] x={best_x}, y={best_y} -> "
          f"Duration={s_best['Duration']}, Finished={s_best['Finished']}, "
          f"Cost={s_best['Total_Undiscounted_Cost']:.3e}")


if __name__ == "__main__":
    main()
