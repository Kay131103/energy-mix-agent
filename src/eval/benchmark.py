"""Held-out benchmark evaluation — base vs SFT vs SFT+GRPO on the identical holdout.

Reports every number honestly (including JSON-parse failures) to results/metrics.json
and results/report.md. Runs on GPU or CPU (0.5B model fits on CPU).
Usage:
  python src/eval/benchmark.py --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow direct-script runs

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.validator.dispatch_solver import Scenario, solve_plan, validate_plan
from src.validator.bill_validator import validate_bill


def generate(model, tok, messages, max_new_tokens=768):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_json(text: str) -> str:
    """Best-effort: return the raw completion (validators try json.loads)."""
    a, b = text.find("{"), text.rfind("}")
    return text[a:b + 1] if (a != -1 and b > a) else text


def eval_rows(model, tok, rows, limit=None, max_new_tokens=768) -> dict:
    res = {"n": 0, "json_ok": 0, "reward_sum": 0.0, "rewards": [], "cost_gaps": []}
    for row in rows[:limit or len(rows)]:
        res["n"] += 1
        t0 = time.time()
        comp = extract_json(generate(model, tok, row["prompt"], max_new_tokens))
        if row["task"] == "plan":
            scen = Scenario.from_dict(row["scenario"])
            opt = solve_plan(scen)
            r = validate_plan(scen, comp, optimal_cost=opt.cost)
            if r["cost_gap_pct"] is not None:
                res["cost_gaps"].append(r["cost_gap_pct"])
        else:
            r = validate_bill(comp, row["truth"])
        res["rewards"].append(r["reward"])
        res["reward_sum"] += r["reward"]
        if "unparseable_json" not in r["violations"]:
            res["json_ok"] += 1
        row.setdefault("_timing", []).append(round(time.time() - t0, 2))
    n = res["n"] or 1
    res["mean_reward"] = round(res["reward_sum"] / n, 4)
    res["json_parse_rate"] = round(res["json_ok"] / n, 4)
    res["mean_cost_gap_pct"] = (round(sum(res["cost_gaps"]) / len(res["cost_gaps"]), 2)
                                if res["cost_gaps"] else None)
    return res


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    max_new = int(cfg.get("max_new_tokens", 768))
    rows = [json.loads(l) for l in open(cfg["holdout_file"], encoding="utf-8")]
    plans = [r for r in rows if r["task"] == "plan"]
    bills = [r for r in rows if r["task"] == "bill"]

    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": cfg["seed"],
           "holdout": {"plan": len(plans), "bill": len(bills)}, "models": {}}

    for tag, adapter in cfg["models"].items():
        tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.float32)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.eval()
        m = {"plan": eval_rows(model, tok, plans, cfg.get("limit"), max_new),
             "bill": eval_rows(model, tok, bills, cfg.get("limit"), max_new)}
        out["models"][tag] = m
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path("results").mkdir(exist_ok=True)
    Path("results", "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# Benchmark results", "", "| model | plan reward | plan JSON% | plan cost gap vs LP optimum | bill reward | bill field JSON% |",
             "|---|---|---|---|---|---|"]
    for tag, m in out["models"].items():
        lines.append(
            f"| {tag} | {m['plan']['mean_reward']} | {m['plan']['json_parse_rate']:.0%} | "
            f"{m['plan']['mean_cost_gap_pct']}% | {m['bill']['mean_reward']} | {m['bill']['json_parse_rate']:.0%} |")
    lines += ["", "Every number is produced by `src/eval/benchmark.py` on the identical holdout "
              "(`data/benchmark/holdout.jsonl`). Raw values in `results/metrics.json`."]
    Path("results", "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    main(args.config)
