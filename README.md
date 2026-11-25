# Energy-Mix Agent — verifiable-reward post-training (LoRA SFT + GRPO)

> **Clem-mini**: a small open model post-trained to plan a data center's electricity
> procurement, where a **program (LP solver + validator) scores every answer** —
> mirroring SustainabilityEconomics.ai's "a program can verify the result" philosophy.

**What it does:** given a 12-hour site load curve, time-of-use grid tariffs, solar/wind PPA
profiles and a battery, the model completes an energy-mix plan (grid / solar / wind /
battery-charge / battery-discharge per hour). A **SciPy LP solver** computes the true optimal
plan; the model is scored on *cost gap vs. optimum* and *hard-constraint feasibility* —
never by an LLM judge. A second task (electricity-bill extraction) is verified arithmetically.

## Pipeline

```
synthetic data (scenarios, bills, solver-optimal plans)
        │
        ▼
 Stage 1 — LoRA SFT            Stage 2 — GRPO (verifiable rewards)
 Qwen2.5-0.5B-Instruct         reward = validator score:
 + PEFT/TRL                    feasibility(0/1) · (1 - cost_gap/25%)
 solver plans as demos  ───►   model improves against the LP optimum
        │                              │
        ▼                              ▼
 held-out benchmark eval — reported honestly (JSON + markdown table)
```

**Key idea from the job description, implemented:** *"Solvers, calculators and simulators
perform the arithmetic, and the models are trained to use them correctly."* Here the LP
solver is both the **SFT teacher** (its optimal plans are the demonstrations — a form of
solver→model distillation) and the **RL reward function** (programmatic grader).

## Repository map

```
configs/                     reproducible run configs (seed, LoRA rank, hyperparams)
data/                        generated datasets (committed; regenerate with seed 42)
  sft/   train.jsonl, val.jsonl          chat-format messages
  grpo/  prompts.jsonl                   prompt + scenario + solver reference
  benchmark/ holdout.jsonl               unseen scenarios + bills for final eval
notebooks/                   2_click_run notebooks for free Colab T4
results/                     demo validator output + (after training) metrics
scripts/                     local, CPU-only: data gen, solver demo, tests
src/
  data/generate_synthetic.py scenario/bill generator + solver-plan demos
  validator/dispatch_solver.py  LP formulation, solve, validate_plan() grader
  validator/bill_validator.py   arithmetic grader for bill extraction
  validator/rewards.py          GRPO reward functions (parse JSON → score)
  sft/train_sft.py              LoRA SFT (TRL SFTTrainer + PEFT)
  rl/train_grpo.py              GRPO (TRL GRPOTrainer, validator rewards)
  eval/benchmark.py             held-out eval → results/metrics.json + report.md
tests/test_validator.py        unit tests for solver + graders
```

## Run it (all free)

| Step | Where | Cost |
|---|---|---|
| Generate data, run solver demo, tests | any CPU | free |
| Stage 1 SFT | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/01_train_sft_colab.ipynb) free T4 | free |
| Stage 2 GRPO | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/02_train_grpo_colab.ipynb) free T4 | free |
| Benchmark eval | Colab or local CPU (0.5B runs on CPU) | free |

```bash
# local sanity check (no GPU needed)
pip install -r requirements-local.txt
python -m src.data.generate_synthetic --seed 42
python -m scripts.demo_validator
pytest tests/ -q
```

Colab: open a notebook, **Runtime → T4 GPU**, Run all. Adapters land in `runs/`;
download them at the end of each notebook.

## Design details

- **Task A — energy-mix planning** (flagship): 12-hour horizon; hard constraints =
  hourly demand balance, solar/wind availability caps, battery power/energy/efficiency
  limits, state-of-charge dynamics, carbon-free share ≥ target. Objective = procurement
  cost. `dispatch_solver.py` formulates this as an LP (72 variables, HiGHS).
- **Task B — bill extraction**: model returns JSON (total kWh, energy charge, fixed
  charge, tax, total due); grader checks field accuracy *and* internal arithmetic.
- **Rewards are programmatic only.** No LLM-as-judge anywhere: GRPO reward =
  `0` on any hard-constraint violation or unparseable JSON, else `1 − min(cost_gap, 25%)/25%`.
- **Honest reporting:** the benchmark evaluates the *base* model, the SFT model and the
  GRPO model on the identical holdout and writes every number to
  `results/metrics.json` — including negative results.

## Results

See `results/report.md` after running the notebooks. Validator demo (local, no training —
full output in `results/demo_output.txt`):

| plan produced by | reward | cost gap vs LP optimum | verdict |
|---|---|---|---|
| LP solver (reference) | 1.00 | ~0% | ₹11,559 vs ₹16,637 grid-only (−30%) |
| greedy heuristic | 0.83 | 4.35% | feasible, but pays a premium |
| corrupted JSON | 0.00 | — | grader rejects with exact violation list |

Bill grader: correct bill 1.00, wrong total 0.42 with `arithmetic_total` violation.
Every score is computed by code — no LLM judge anywhere.

## Alignment with the role

| Job scope item | Where in this repo |
|---|---|
| Prepare, clean & verify training data | `src/data/generate_synthetic.py` (seeded, schema-checked) |
| SFT of open models with LoRA | `src/sft/train_sft.py` (TRL + PEFT, Qwen2.5-0.5B-Instruct) |
| RL environments & programmatic reward functions | `src/validator/` (LP solver, graders, `rewards.py`) |
| Evaluate before/after training, report accurately | `src/eval/benchmark.py` → `results/metrics.json` |
| Synthetic data where real examples are limited | scenario/bill generator with configurable distributions |
| Distillation | LP solver → model plan distillation in SFT stage |
| Reproducible experiments (configs, logs) | `configs/*.yaml`, seeded generation, run logs in `runs/` |
| Structured outputs / tool calling | JSON-schema-constrained completions + validators |
| Optimization solvers | SciPy `linprog` (HiGHS) |

## Requirements

- Stage-1/2 training: Colab free tier T4 (16 GB) — LoRA keeps a 0.5B model + GRPO
  rollouts comfortably in memory.
- Everything else runs on CPU.
