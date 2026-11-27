@echo off
rem Full local training pipeline: SFT -> GRPO (bill task) -> benchmark eval.
rem All output goes to runs\local_pipeline.log. Safe to re-run: finished stages
rem are skipped (adapter folders exist), interrupted stages resume from checkpoints.
cd /d "%~dp0.."
if not exist runs mkdir runs

echo === [1/3] LoRA SFT ===
if exist runs\sft_local\adapter (
    echo SFT already complete - skipping
) else (
    python src\sft\train_sft.py --config configs\sft_local.yaml >> runs\local_pipeline.log 2>&1
    if errorlevel 1 goto :fail
)

echo === [2/3] GRPO on bill task ===
if exist runs\grpo_local\adapter (
    echo GRPO already complete - skipping
) else (
    python src\rl\train_grpo.py --config configs\grpo_local.yaml >> runs\local_pipeline.log 2>&1
    if errorlevel 1 goto :fail
)

echo === [3/3] Benchmark eval ===
python src\eval\benchmark.py --config configs\eval_local.yaml >> runs\local_pipeline.log 2>&1
if errorlevel 1 goto :fail

echo PIPELINE DONE - see runs\local_pipeline.log and results\report.md
exit /b 0

:fail
echo PIPELINE FAILED - see runs\local_pipeline.log
exit /b 1
