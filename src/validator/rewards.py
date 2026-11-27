"""GRPO reward functions.

TRL GRPOTrainer calls each reward function with (prompts, completions, **kwargs)
where kwargs contain the extra dataset columns (scenario, truth, reference, bill_text).
"""
from __future__ import annotations

import json

from src.validator.dispatch_solver import Scenario, solve_plan, validate_plan
from src.validator.bill_validator import validate_bill


def _as_text(completion) -> str:
    """TRL passes completions as strings (standard format) or as chat-message
    dicts (conversational format). Normalize to the raw text either way."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def plan_reward(prompts, completions, scenario=None, reference=None, **kw) -> list[float]:
    out = []
    for i, comp in enumerate(completions):
        scen = Scenario.from_dict(scenario[i])
        # recompute the LP optimum for this scenario (fast: 72 vars) — the grader and
        # the SFT teacher share this one source of truth
        opt_cost = solve_plan(scen).cost
        out.append(validate_plan(scen, _as_text(comp), optimal_cost=opt_cost)["reward"])
    return out


def bill_reward(prompts, completions, truth=None, **kw) -> list[float]:
    return [validate_bill(_as_text(c), truth[i])["reward"] for i, c in enumerate(completions)]


REWARD_FUNCS = [plan_reward, bill_reward]
