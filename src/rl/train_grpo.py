"""Stage 2 — GRPO with programmatic validator rewards (TRL GRPOTrainer).

Only the plan task is RL-trained (the flagship task); the bill task stays SFT-only.
Usage:
  python src/rl/train_grpo.py --config configs/grpo.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow direct-script runs

import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.validator.rewards import plan_reward


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    rows = [json.loads(l) for l in open(cfg["prompts_file"], encoding="utf-8")]
    task = cfg.get("task", "plan")
    if task == "plan":
        rows = [r for r in rows if "scenario" in r]  # plan task only
    else:
        rows = [r for r in rows if "truth" in r]     # bill task only
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    # CPU-safe dtype: bf16 needs AVX512; fall back to float32 when no GPU
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=dtype, attn_implementation="sdpa")
    if cfg.get("adapter_dir"):
        model = PeftModel.from_pretrained(model, cfg["adapter_dir"])
        model = model.merge_and_unload()

    lora = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM",
    )

    grpo_cfg = GRPOConfig(
        output_dir=cfg["output_dir"], seed=cfg["seed"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_generations=cfg["num_generations"],
        max_completion_length=cfg["max_completion_tokens"],
        max_prompt_length=cfg["max_prompt_tokens"],
        temperature=cfg["temperature"],
        beta=cfg["beta"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg.get("max_steps", -1),
        logging_steps=5, save_strategy="steps", save_steps=cfg["save_steps"],
        save_total_limit=2, bf16=torch.cuda.is_available(), report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = GRPOTrainer(
        model=model, args=grpo_cfg, train_dataset=ds,
        reward_funcs=plan_reward if task == "plan" else bill_reward,
        processing_class=tok, peft_config=lora,
    )
    out = Path(cfg["output_dir"])
    has_ckpt = bool(list(out.glob("checkpoint-*"))) if out.exists() else False
    if cfg.get("resume", False) and has_ckpt:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    final = Path(cfg["output_dir"]) / "adapter"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    Path(cfg["output_dir"], "run_summary.json").write_text(
        json.dumps({"config": cfg, "log_history": trainer.state.log_history[-10:],
                    "train_samples": len(ds)}, indent=2, default=str), encoding="utf-8")
    print(f"adapter saved to {final}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    main(args.config)
