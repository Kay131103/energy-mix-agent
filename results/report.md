# Benchmark results

| model | plan reward | plan JSON% | plan cost gap vs LP optimum | bill reward | bill field JSON% |
|---|---|---|---|---|---|
| base | 0.0 | 0% | None% | 0.8333 | 100% |
| sft | 0.0 | 0% | None% | 0.9889 | 100% |
| sft_grpo | 0.0 | 0% | None% | 0.8333 | 100% |

Every number is produced by `src/eval/benchmark.py` on the identical holdout (`data/benchmark/holdout.jsonl`). Raw values in `results/metrics.json`.