"""Synthetic data generation: energy scenarios, bills, solver-optimal plan demos.

Everything is seeded so the datasets are exactly reproducible.
Outputs (all JSONL):
  data/sft/train.jsonl        chat-format {"messages": [...]}
  data/sft/val.jsonl
  data/grpo/prompts.jsonl     {"prompt": [...], "scenario": {...}, "reference": {...}}
  data/benchmark/holdout.jsonl  mixed tasks, unseen seeds
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.tasks.prompts import bill_prompt, plan_prompt
from src.validator.dispatch_solver import Scenario, solve_plan, plan_to_json

RNG = np.random.default_rng

HOURS = 12  # 12-hour rolling procurement window


def make_scenario(rng) -> Scenario:
    """Builds a scenario and returns it only if the LP is feasible (deterministic per seed)."""
    load = (180 + rng.integers(-30, 31, HOURS) + np.linspace(0, 40, HOURS)).round(1)
    # time-of-use: cheap night, expensive evening peak
    rate = np.where(
        np.arange(HOURS) < 6,
        rng.uniform(4.0, 5.5),
        np.where(np.arange(HOURS) < 10, rng.uniform(6.5, 8.5), rng.uniform(9.0, 12.5)),
    ).round(2)
    for _attempt in range(100):
        scen = Scenario(
            scenario_id=f"sc_{rng.integers(10**6):06d}",
            load_kwh=load.tolist(),
            grid_tou_rate=rate.tolist(),
            grid_cap_kwh=float(rng.uniform(1.0, 1.4) * load.max()),
            solar_ppa_price=float(rng.uniform(2.6, 3.8)),
            solar_cap_kw=float(rng.uniform(150, 300)),
            solar_avail_profile=np.clip(
                rng.uniform(-0.1, 1.15, HOURS) * (0.9 - abs(np.arange(HOURS) - 5.5) / 7), 0, 1
            ).round(2).tolist(),
            wind_ppa_price=float(rng.uniform(3.8, 5.2)),
            wind_cap_kw=float(rng.uniform(100, 220)),
            wind_avail_profile=np.clip(rng.uniform(0.15, 1.0, HOURS), 0, 1).round(2).tolist(),
            battery_power_kw=float(rng.uniform(20, 60)),
            battery_energy_kwh=float(rng.uniform(120, 300)),
            battery_charge_eff=float(rng.uniform(0.9, 0.97)),
            battery_discharge_eff=float(rng.uniform(0.9, 0.97)),
            battery_initial_soc_kwh=float(rng.uniform(0.3, 0.6) * 150),
            battery_opex_per_kwh=float(rng.uniform(0.1, 0.35)),
            carbon_target_share=float(rng.uniform(0.5, 0.75)),
        )
        if solve_plan(scen).feasible:
            return scen
    raise RuntimeError("no feasible scenario found in 100 attempts")


def make_bill(rng) -> dict:
    """Generate a synthetic electricity bill + its ground-truth extraction."""
    tax_pct = float(rng.choice([5.0, 12.0, 18.0]))
    fixed = float(round(rng.uniform(400, 1500), 0))
    n_buckets = int(rng.integers(2, 4))
    kwh = rng.integers(1500, 6000, n_buckets).astype(float)
    rate = rng.uniform(5.5, 11.5, n_buckets).round(2)
    lines = [
        {"bucket": ["off-peak", "normal", "peak"][i % 3], "kwh": float(kwh[i]), "rate": float(rate[i])}
        for i in range(n_buckets)
    ]
    energy_charge = float(round(sum(l["kwh"] * l["rate"] for l in lines), 2))
    tax = float(round(energy_charge * tax_pct / 100, 2))
    total = float(round(energy_charge + fixed + tax, 2))
    truth = {
        "period": f"2026-{rng.integers(1, 13):02d}",
        "total_kwh": float(kwh.sum()),
        "energy_charge": energy_charge,
        "fixed_charge": fixed,
        "tax": tax,
        "total_due": total,
    }
    text = [f"COMMERCIAL ELECTRICITY BILL — site SIT-{rng.integers(1000, 9999)}",
            f"Billing period: {truth['period']}", "Energy charges:"]
    for l in lines:
        text.append(f"  {l['bucket']:<9} {l['kwh']:>8.0f} kWh  @ ₹{l['rate']:.2f}/kWh  = ₹{l['kwh']*l['rate']:.2f}")
    text += [f"Energy charge total: ₹{energy_charge:.2f}",
             f"Fixed charge: ₹{fixed:.2f}",
             f"Tax @ {tax_pct:.0f}%: ₹{tax:.2f}",
             f"TOTAL DUE: ₹{total:.2f}"]
    return {"bill_text": "\n".join(text), "truth": truth}


SYSTEM_PLAN = (
    "You are an energy-procurement planner. Complete the JSON energy-mix plan for the "
    "site. Reply with ONLY JSON: {\"hours\":[{\"h\":int,\"grid\":num,\"solar\":num,"
    "\"wind\":num,\"chg\":num,\"dis\":num,\"soc\":num}, ...]} for all 12 hours. "
    "grid+solar+wind+dis-chg must equal the hourly load; obey all caps, battery "
    "state-of-charge dynamics and the carbon-free share target."
)
SYSTEM_BILL = (
    "You are a utility-bill extractor. Reply with ONLY JSON with keys: period (str), "
    "total_kwh (num), energy_charge (num), fixed_charge (num), tax (num), total_due (num)."
)


def chat(system: str, user: str, assistant: str | None = None) -> dict:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


def main(seed: int = 42, n_train: int = 900, n_val: int = 80, n_grpo: int = 400, n_holdout: int = 140):
    rng = RNG(seed)
    base = Path("data")
    for sub in ["sft", "grpo", "benchmark"]:
        (base / sub).mkdir(parents=True, exist_ok=True)

    # ---------- Task A: plans (SFT demos = LP-optimal plans from the solver) ----------
    plan_train, plan_val = [], []
    while len(plan_train) + len(plan_val) < n_train + n_val:
        scen = make_scenario(rng)
        sol = solve_plan(scen)
        if not sol.feasible:  # LP failure guard (should never trigger)
            continue
        target = plan_to_json(scen, sol)
        row = chat(SYSTEM_PLAN, plan_prompt(scen), target)
        (plan_train if len(plan_train) < n_train else plan_val).append(row)

    # ---------- Task B: bills ----------
    bill_train, bill_val = [], []
    for _ in range((n_train + n_val) // 3):
        b = make_bill(rng)
        row = chat(SYSTEM_BILL, bill_prompt(b["bill_text"]),
                   json.dumps(b["truth"], separators=(",", ":")))
        (bill_train if len(bill_train) < (n_train * 2) // 3 else bill_val).append(row)

    # ---------- write SFT ----------
    train = plan_train + bill_train
    val = plan_val + bill_val
    rng.shuffle(train); rng.shuffle(val)
    for name, rows in [("train", train), ("val", val)]:
        with open(base / "sft" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # ---------- GRPO prompts (fresh scenarios, keep scenario+reference as columns) ----
    with open(base / "grpo" / "prompts.jsonl", "w", encoding="utf-8") as f:
        for _ in range(n_grpo // 2):
            scen = make_scenario(rng)
            sol = solve_plan(scen)
            f.write(json.dumps({
                "prompt": [{"role": "system", "content": SYSTEM_PLAN},
                           {"role": "user", "content": plan_prompt(scen)}],
                "scenario": scen.to_dict(),
                "reference": plan_to_json(scen, sol),
            }) + "\n")
        for _ in range(n_grpo // 2):
            b = make_bill(rng)
            f.write(json.dumps({
                "prompt": [{"role": "system", "content": SYSTEM_BILL},
                           {"role": "user", "content": bill_prompt(b["bill_text"])}],
                "bill_text": b["bill_text"],
                "truth": b["truth"],
            }) + "\n")

    # ---------- benchmark holdout ----------
    with open(base / "benchmark" / "holdout.jsonl", "w", encoding="utf-8") as f:
        for _ in range(n_holdout // 2):
            scen = make_scenario(rng)
            sol = solve_plan(scen)
            f.write(json.dumps({
                "task": "plan",
                "prompt": [{"role": "system", "content": SYSTEM_PLAN},
                           {"role": "user", "content": plan_prompt(scen)}],
                "scenario": scen.to_dict(),
                "reference": plan_to_json(scen, sol),
            }) + "\n")
        for _ in range(n_holdout // 2):
            b = make_bill(rng)
            f.write(json.dumps({
                "task": "bill",
                "prompt": [{"role": "system", "content": SYSTEM_BILL},
                           {"role": "user", "content": bill_prompt(b["bill_text"])}],
                "bill_text": b["bill_text"],
                "truth": b["truth"],
            }) + "\n")

    print(f"SFT: {len(train)} train / {len(val)} val | GRPO: {n_grpo} prompts | "
          f"benchmark: {n_holdout}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(seed=args.seed)
