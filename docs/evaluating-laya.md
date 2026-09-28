# Comparing AnyJev with Laya

Use the same `LocalLLaMA/typed-decisions` test split for both runs. The Laya runner measures a
Laya checkpoint as a baseline. `bench.run_typed` measures an open base model with AnyJev's raw,
L0, and L1 decision levels.

## Install

From the repository root, install AnyJev's model and benchmark dependencies, then install Laya for
the baseline run:

```bash
pip install -e ".[hf,bench]"
pip install laya
```

The commands download the dataset and checkpoints on first run. Choose a model and device that fit
your machine.

## Run the Laya baseline

```bash
python -m bench.providers.laya \
  --checkpoint convaiinnovations/laya-typed-decisions \
  --device cuda
```

Use `convaiinnovations/laya` for the zero-shot Laya checkpoint. The runner writes a JSON result
under `bench/results_typed/<date>/`. Set `--device cpu` when CUDA is unavailable, or use
`--limit-cases 5` for a small smoke run. A limited run is not comparable to the full results.

## Run AnyJev on an open model

```bash
python -m bench.run_typed \
  --model Qwen/Qwen3-8B \
  --levels raw,L0,L1 \
  --calib-cases 200
```

The runner evaluates the test split and uses only the train split to fit L1 temperature scaling.
Its JSON and Markdown output go under `bench/results_typed/<date>/`. For a quick smoke run, add
`--limit-cases 5 --calib-cases 5`; this limits cases per workflow and should not be compared with
the full-run rows.

Run both commands on the same day to place their JSON files in the same dated directory. Then print
the combined table with `python -m bench.typed_table bench/results_typed/<date>`. The table's
protocol and existing results are described in [results_typed.md](results_typed.md). To regenerate
all result documents from the latest files, use the repository's full command:
`bash scripts/regen_docs.sh bench/results_v01 bench/results_typed bench/results_typed`.

## What “AnyJev over Laya” means here

The current Laya provider is a baseline runner. It does not wrap Laya's predictions in AnyJev. To
measure AnyJev, use a base model supported by `HFBackend` with `bench.run_typed`. A Laya-backed
AnyJev backend would be a separate integration.
