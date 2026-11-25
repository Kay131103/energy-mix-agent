import json

import numpy as np
import pytest

from src.data.generate_synthetic import RNG, make_scenario, make_bill
from src.validator.dispatch_solver import Scenario, solve_plan, validate_plan, grid_only_cost
from src.validator.bill_validator import validate_bill


def test_solver_feasible_and_beats_or_matches_grid_only():
    scen = make_scenario(RNG(0))
    sol = solve_plan(scen)
    assert sol.feasible
    assert sol.cost <= grid_only_cost(scen) + 1e-6


def test_solver_plan_scores_full_reward():
    scen = make_scenario(RNG(1))
    sol = solve_plan(scen)
    comp = json.dumps(sol.plan, separators=(",", ":"))
    r = validate_plan(scen, comp, optimal_cost=sol.cost)
    assert r["violations"] == []
    assert r["reward"] >= 0.99  # cost gap ~0 for the reference plan


def test_validator_catches_violations():
    scen = make_scenario(RNG(2))
    sol = solve_plan(scen)
    plan = json.loads(json.dumps(sol.plan))
    plan["hours"][3]["grid"] += 500  # break balance & possibly carbon cap
    r = validate_plan(scen, json.dumps(plan), optimal_cost=sol.cost)
    assert r["reward"] == 0.0 and r["violations"]


def test_unparseable_completion_scores_zero():
    scen = make_scenario(RNG(3))
    assert validate_plan(scen, "I think the answer is maybe", 100.0)["reward"] == 0.0


def test_bill_grader():
    b = make_bill(RNG(4))
    good = validate_bill(json.dumps(b["truth"]), b["truth"])
    assert good["reward"] >= 0.99
    bad = validate_bill(json.dumps({**b["truth"], "total_due": 1.0}), b["truth"])
    assert bad["reward"] < 0.99
    assert validate_bill("not json", b["truth"])["reward"] == 0.0


def test_scenarios_round_trip():
    scen = make_scenario(RNG(5))
    assert Scenario.from_dict(scen.to_dict()).to_dict() == scen.to_dict()
