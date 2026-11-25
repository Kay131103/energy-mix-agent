"""GRPO reward functions.

TRL GRPOTrainer calls each reward function with (prompts, completions, **kwargs)
where kwargs contain the extra dataset columns (scenario, truth, reference, bill_text).
"""
from __future__ import annotations

import json

from src.validator.dispatch_solver import Scenario, validate_plan
from src.validator.bill_validator import validate_bill


def plan_reward(prompts, completions, scenario=None, reference=None, **kw) -> list[float]:
    out = []
    for i, comp in enumerate(completions):
        scen = Scenario.from_dict(scenario[i])
        # recompute the LP optimum for this scenario (fast: 72 vars) — the grader and
        # the SFT teacher share this one source of truth
        opt_cost = solve_plan(scen).cost
        out.append(validate_plan(scen, comp, optimal_cost=opt_cost)["reward"])
    return out


def bill_reward(prompts, completions, truth=None, **kw) -> list[float]:
    return [validate_bill(c, truth[i])["reward"] for i, c in enumerate(completions)]


REWARD_FUNCS = [plan_reward, bill_reward]
