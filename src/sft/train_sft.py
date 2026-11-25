"""Stage 1 — LoRA supervised fine-tuning of Qwen2.5-0.5B-Instruct on solver demos.

Runs on a free Colab T4. Usage:
  python src/sft/train_sft.py --config configs/sft.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    random.seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    ds = load_dataset("json", data_files={
        "train": cfg["train_file"], "validation": cfg["val_file"]})

    lora = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        per_device_eval_batch_size=cfg["batch_size"],
        eval_strategy="steps", eval_steps=cfg["eval_steps"],
        save_strategy="steps", save_steps=cfg["eval_steps"],
        save_total_limit=2, logging_steps=10, bf16=True, seed=cfg["seed"],
        max_length=cfg["max_seq_len"], packing=False,
        report_to=[],  # keep runs fully local & reproducible
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds["train"],
                         eval_dataset=ds["validation"], processing_class=tok,
                         peft_config=lora)
    trainer.train()

    final = Path(cfg["output_dir"]) / "adapter"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    # write a small run summary (reproducibility: config + metrics snapshot)
    summary = {"config": cfg, "train_samples": len(ds["train"]),
               "log_history": trainer.state.log_history[-5:]}
    Path(cfg["output_dir"], "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"adapter saved to {final}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    args = ap.parse_args()
    main(args.config)
