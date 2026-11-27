"""LP formulation of the 12-hour energy-mix procurement problem + plan validator/grader.

The same module is used three ways (by design, mirroring the job description):
  1. solve_plan()          — SciPy LP produces the *reference* plan (SFT teacher).
  2. plan_to_json()        — converts the LP solution into the model's target format.
  3. validate_plan()       — programmatic grader / GRPO reward for model completions.

Decision variables (72): per hour h: grid g[h], solar s[h], wind w[h],
battery charge c[h], discharge d[h], state-of-charge soc[h].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.optimize import linprog


@dataclass
class Scenario:
    scenario_id: str
    load_kwh: list
    grid_tou_rate: list
    grid_cap_kwh: float
    solar_ppa_price: float
    solar_cap_kw: float
    solar_avail_profile: list
    wind_ppa_price: float
    wind_cap_kw: float
    wind_avail_profile: list
    battery_power_kw: float
    battery_energy_kwh: float
    battery_charge_eff: float
    battery_discharge_eff: float
    battery_initial_soc_kwh: float
    battery_opex_per_kwh: float
    carbon_target_share: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(**d)


@dataclass
class Solution:
    feasible: bool
    cost: float = float("nan")
    grid_only_cost: float = float("nan")
    carbon_free_share: float = float("nan")
    plan: dict = field(default_factory=dict)


def _h(scen: Scenario) -> int:
    return len(scen.load_kwh)


def grid_only_cost(scen: Scenario) -> float:
    return float(np.dot(scen.load_kwh, scen.grid_tou_rate))


def solve_plan(scen: Scenario) -> Solution:
    H = _h(scen)
    n = 6 * H
    V = lambda name, h: name * H + h  # variable layout

    c_obj = np.zeros(n)
    for h in range(H):
        c_obj[V(0, h)] = scen.grid_tou_rate[h]           # grid
        c_obj[V(1, h)] = scen.solar_ppa_price            # solar
        c_obj[V(2, h)] = scen.wind_ppa_price             # wind
        c_obj[V(3, h)] = scen.battery_opex_per_kwh       # charge opex
        c_obj[V(4, h)] = scen.battery_opex_per_kwh       # discharge opex
        # soc has zero cost

    A_eq, b_eq = [], []
    for h in range(H):
        row = np.zeros(n)
        row[V(0, h)] = 1; row[V(1, h)] = 1; row[V(2, h)] = 1
        row[V(4, h)] = 1; row[V(3, h)] = -1              # g+s+w+d-c = load
        A_eq.append(row); b_eq.append(scen.load_kwh[h])
    for h in range(H):  # soc dynamics: soc[h] = soc[h-1] + eta_c*c[h] - d[h]/eta_d
        row = np.zeros(n)
        row[V(5, h)] = 1; row[V(3, h)] = -scen.battery_charge_eff
        row[V(4, h)] = 1 / scen.battery_discharge_eff
        if h == 0:
            b_eq.append(scen.battery_initial_soc_kwh)
        else:
            row[V(5, h - 1)] = -1
            b_eq.append(0.0)
        A_eq.append(row)

    # carbon-free share: sum(g) <= (1 - target) * sum(load)
    row = np.zeros(n)
    for h in range(H):
        row[V(0, h)] = 1.0
    A_ub = [row]
    b_ub = [(1 - scen.carbon_target_share) * sum(scen.load_kwh)]
    # end-of-window SoC must not be lower than start (no free energy mining)
    row = np.zeros(n); row[V(5, H - 1)] = -1.0
    A_ub.append(row); b_ub.append(-scen.battery_initial_soc_kwh)

    # Block-major bounds matching the V(name, h) layout:
    # [g0..g11 | s0..s11 | w0..w11 | c0..c11 | d0..d11 | soc0..soc11]
    bounds = (
        [(0, scen.grid_cap_kwh) for _ in range(H)]
        + [(0, scen.solar_avail_profile[h] * scen.solar_cap_kw) for h in range(H)]
        + [(0, scen.wind_avail_profile[h] * scen.wind_cap_kw) for h in range(H)]
        + [(0, scen.battery_power_kw) for _ in range(H)]
        + [(0, scen.battery_power_kw) for _ in range(H)]
        + [(0, scen.battery_energy_kwh) for _ in range(H)]
    )

    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if not res.success:
        return Solution(feasible=False)
    x = res.x
    plan = {"hours": [{"h": h,
                       "grid": round(float(x[V(0, h)]), 2),
                       "solar": round(float(x[V(1, h)]), 2),
                       "wind": round(float(x[V(2, h)]), 2),
                       "chg": round(float(x[V(3, h)]), 2),
                       "dis": round(float(x[V(4, h)]), 2),
                       "soc": round(float(x[V(5, h)]), 2)} for h in range(H)]}
    soc0 = scen.battery_initial_soc_kwh
    prev, socs = soc0, []
    for hh in plan["hours"]:  # reconstruct SoC exactly as the validator will
        prev = prev + scen.battery_charge_eff * hh["chg"] - hh["dis"] / scen.battery_discharge_eff
        socs.append(round(prev, 2)); hh["soc"] = round(prev, 2)
    return Solution(feasible=True, cost=float(res.fun), grid_only_cost=grid_only_cost(scen),
                    carbon_free_share=1.0 - float(x[:H].sum()) / sum(scen.load_kwh), plan=plan)


def plan_to_json(scen: Scenario, sol: Solution) -> str:
    return json.dumps(sol.plan, separators=(",", ":"))


# --------------------------------------------------------------------------
# Grader: score a *model-produced* plan against the scenario constraints.
# --------------------------------------------------------------------------

BALANCE_TOL = 1.5      # kWh, hourly demand balance tolerance
DYN_TOL = 1.5          # kWh, SoC dynamics tolerance
CAP_TOL = 0.5          # kWh, cap tolerance
MAX_GAP_PCT = 25.0     # cost gap that maps to reward 0


def extract_json(text: str) -> str:
    """Tolerant extraction: strip markdown fences / prose, keep the outermost {...}."""
    a, b = text.find("{"), text.rfind("}")
    return text[a:b + 1] if (a != -1 and b > a) else text


def validate_plan(scen: Scenario, completion: str, optimal_cost: float | None = None) -> dict:
    """Programmatic grader. Returns {"violations": [...], "cost": ..., "reward": float}."""
    v: list[str] = []
    try:
        plan = json.loads(extract_json(completion))
        hours = plan["hours"]
        assert isinstance(hours, list) and len(hours) == _h(scen)
    except Exception:
        return {"violations": ["unparseable_json"], "cost": None, "reward": 0.0,
                "carbon_free_share": None, "cost_gap_pct": None}

    soc = scen.battery_initial_soc_kwh
    tot_grid = 0.0
    for i, hh in enumerate(hours):
        try:
            g, s, w, c, d = (float(hh["grid"]), float(hh["solar"]), float(hh["wind"]),
                             float(hh["chg"]), float(hh["dis"]))
        except Exception:
            return {"violations": [f"hour_{i}:missing_field"], "cost": None, "reward": 0.0,
                    "carbon_free_share": None, "cost_gap_pct": None}
        if any(x < -CAP_TOL for x in (g, s, w, c, d)):
            v.append(f"hour_{i}:negative")
        if abs(g + s + w + d - c - scen.load_kwh[i]) > BALANCE_TOL:
            v.append(f"hour_{i}:balance")
        if g > scen.grid_cap_kwh + CAP_TOL:
            v.append(f"hour_{i}:grid_cap")
        if s > scen.solar_avail_profile[i] * scen.solar_cap_kw + CAP_TOL:
            v.append(f"hour_{i}:solar_cap")
        if w > scen.wind_avail_profile[i] * scen.wind_cap_kw + CAP_TOL:
            v.append(f"hour_{i}:wind_cap")
        if c > scen.battery_power_kw + CAP_TOL or d > scen.battery_power_kw + CAP_TOL:
            v.append(f"hour_{i}:battery_power")
        if c > CAP_TOL and d > CAP_TOL:
            v.append(f"hour_{i}:simultaneous_chg_dis")
        soc = soc + scen.battery_charge_eff * c - d / scen.battery_discharge_eff
        if soc < -CAP_TOL or soc > scen.battery_energy_kwh + CAP_TOL:
            v.append(f"hour_{i}:soc_bounds")
        tot_grid += max(g, 0)

    if tot_grid > (1 - scen.carbon_target_share) * sum(scen.load_kwh) + 1.0:
        v.append("carbon_share")
    if soc < scen.battery_initial_soc_kwh - DYN_TOL:
        v.append("end_soc_below_start")

    cost = None
    gap = None
    if not v:
        cost = 0.0
        for i, hh in enumerate(hours):
            cost += (float(hh["grid"]) * scen.grid_tou_rate[i] + float(hh["solar"]) * scen.solar_ppa_price
                     + float(hh["wind"]) * scen.wind_ppa_price
                     + scen.battery_opex_per_kwh * (float(hh["chg"]) + float(hh["dis"])))
        ref = optimal_cost if optimal_cost is not None else grid_only_cost(scen)
        gap = max(0.0, (cost - ref) / ref * 100.0) if ref > 0 else 0.0
    reward = 0.0 if v else 1.0 - min(gap or 0.0, MAX_GAP_PCT) / MAX_GAP_PCT
    share = 1.0 - tot_grid / sum(scen.load_kwh) if v == [] else None
    return {"violations": v, "cost": cost, "cost_gap_pct": gap,
            "carbon_free_share": share, "reward": round(reward, 4)}
