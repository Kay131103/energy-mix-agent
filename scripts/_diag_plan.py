"""Diagnose plan-task failure: what does the SFT model actually emit for plans?"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.validator.dispatch_solver import Scenario, validate_plan

rows = [json.loads(l) for l in open("data/benchmark/holdout.jsonl", encoding="utf-8")
        if json.loads(l)["task"] == "plan"][:2]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, "runs/sft_local/adapter").merge_and_unload()
model.eval()

for i, row in enumerate(rows):
    prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    n_prompt_tokens = len(tok(prompt)["input_ids"])
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=800, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    comp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    n_comp_tokens = out.shape[1] - ids["input_ids"].shape[1]
    scen = Scenario.from_dict(row["scenario"])
    r = validate_plan(scen, comp, optimal_cost=2000.0)
    print(f"--- plan {i}: prompt={n_prompt_tokens} tok, completion={n_comp_tokens} tok, reward={r['reward']}")
    print(repr(comp[:400]))
