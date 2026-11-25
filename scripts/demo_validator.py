"""Local CPU demo: shows the solver, the validator and the reward working together.

Run: python -m scripts.demo_validator
Output is committed to results/demo_output.txt as evidence the graders are programmatic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.generate_synthetic import RNG, HOURS, make_scenario
from src.validator.dispatch_solver import Scenario, solve_plan, validate_plan, grid_only_cost
from src.validator.bill_validator import validate_bill


def greedy_plan(scen: Scenario) -> str:
    """Deliberately naive heuristic: renewables-first, battery arbitraged simply."""
    H = HOURS
    hours = []
    soc = scen.battery_initial_soc_kwh
    peak_hours = sorted(range(H), key=lambda h: -scen.grid_tou_rate[h])[:4]
    for h in range(H):
        s = min(scen.solar_avail_profile[h] * scen.solar_cap_kw, scen.load_kwh[h])
        w = min(scen.wind_avail_profile[h] * scen.wind_cap_kw, max(scen.load_kwh[h] - s, 0))
        need = scen.load_kwh[h] - s - w
        d = 0.0
        if h in peak_hours and soc > 1:
            d = min(scen.battery_power_kw, max(need, 0) * 0.5, soc * scen.battery_discharge_eff)
        g = max(need - d, 0)
        g = min(g, scen.grid_cap_kwh)
        chg = 0.0
        if scen.grid_tou_rate[h] < np.median(scen.grid_tou_rate) and g < scen.grid_cap_kwh:
            chg = min(scen.battery_power_kw, scen.grid_cap_kwh - g,
                      max(scen.battery_energy_kwh - soc, 0) / scen.battery_charge_eff)
        g = max(scen.load_kwh[h] - s - w - d + chg, 0)  # keep balance exact
        soc = soc + scen.battery_charge_eff * chg - d / scen.battery_discharge_eff
        hours.append({"h": h, "grid": round(g, 2), "solar": round(s, 2), "wind": round(w, 2),
                      "chg": round(chg, 2), "dis": round(d, 2), "soc": round(soc, 2)})
    return json.dumps({"hours": hours}, separators=(",", ":"))


def main(seed: int = 7):
    rng = RNG(seed)
    lines = ["# Validator demo (local CPU, no training involved)", ""]
    for i in range(3):
        scen = make_scenario(rng)
        sol = solve_plan(scen)
        assert sol.feasible
        opt_json = json.dumps(sol.plan, separators=(",", ":"))

        opt = validate_plan(scen, opt_json, optimal_cost=sol.cost)
        greedy = validate_plan(scen, greedy_plan(scen), optimal_cost=sol.cost)
        broken = validate_plan(scen, opt_json.replace('"grid"', '"gridX"'), optimal_cost=sol.cost)

        lines += [f"## Scenario {scen.scenario_id}",
                  f"- grid-only cost: ₹{grid_only_cost(scen):.0f}",
                  f"- LP optimum cost: ₹{sol.cost:.0f} "
                  f"(carbon-free share {sol.carbon_free_share:.0%})",
                  f"- reward(LP plan)          = {opt['reward']}",
                  f"- reward(greedy heuristic) = {greedy['reward']}  "
                  f"(cost gap {greedy['cost_gap_pct']}%, "
                  f"violations: {greedy['violations'] or 'none'})",
                  f"- reward(corrupted JSON)   = {broken['reward']}  "
                  f"(violations: {broken['violations'][:3]}...)", ""]

    bill_truth = {"period": "2026-03", "total_kwh": 4200.0, "energy_charge": 33600.0,
                  "fixed_charge": 900.0, "tax": 6048.0, "total_due": 40548.0}
    good = validate_bill(json.dumps(bill_truth), bill_truth)
    bad = validate_bill(json.dumps({**bill_truth, "total_due": 99999.0}), bill_truth)
    lines += ["## Bill grader",
              f"- reward(correct bill)     = {good['reward']} (field_acc {good['field_acc']})",
              f"- reward(wrong total)      = {bad['reward']} (violations {bad['violations']})",
              "", "Rewards are computed by code only — no LLM judge anywhere."]

    text = "\n".join(lines)
    Path("results").mkdir(exist_ok=True)
    Path("results", "demo_output.txt").write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:  # Windows cp1252 console
        import sys
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
