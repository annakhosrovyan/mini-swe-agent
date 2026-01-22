# Run and Evaluate SWE-bench 🚀

## Run (example) 

```
python -m minisweagent.run.extra.swebench --subset lite --split dev --output runs/openai-lite-bm25 --workers 1 --config src/minisweagent/config/extra/swebench.yaml --retrieval hybrid
```

## Retrieval options

* `none`
* `bm25` (all files)
* `bm25_py` (Python files only)
* `bm25_source` (source code only)
* `bm25_two_stage` (two-stage with entity reranking)
* `hybrid` (chunk-based embedding + BM25)

## Evaluate (example)

```
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-Bench_Lite --split dev --predictions_path runs/openai-lite-bm25/preds.json --run_id multi_test --report_dir runs/eval
```
