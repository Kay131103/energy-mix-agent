"""Arithmetic grader for the bill-extraction task. No LLM judging — pure field checks."""
from __future__ import annotations

import json

FIELDS = ["period", "total_kwh", "energy_charge", "fixed_charge", "tax", "total_due"]
MONEY_TOL = 0.05  # INR

from src.validator.dispatch_solver import extract_json  # shared tolerant parser


def validate_bill(completion: str, truth: dict) -> dict:
    try:
        pred = json.loads(extract_json(completion))
        assert isinstance(pred, dict)
    except Exception:
        return {"violations": ["unparseable_json"], "field_acc": 0.0, "reward": 0.0}

    correct = 0
    for f in FIELDS:
        if f not in pred:
            continue
        t, p = truth[f], pred[f]
        if isinstance(t, (int, float)):
            ok = isinstance(p, (int, float)) and abs(float(p) - float(t)) <= MONEY_TOL
        else:
            ok = str(p).strip() == str(t).strip()
        correct += int(ok)

    v = []
    # internal arithmetic consistency (the "calculator" check)
    try:
        if abs(pred["energy_charge"] * (1 + 0) + pred["fixed_charge"] + pred["tax"]
               - pred["total_due"]) > MONEY_TOL * 2:
            v.append("arithmetic_total")
    except Exception:
        v.append("missing_field")

    field_acc = correct / len(FIELDS)
    reward = field_acc if not v else field_acc * 0.5
    return {"violations": v, "field_acc": round(field_acc, 4), "reward": round(reward, 4)}
