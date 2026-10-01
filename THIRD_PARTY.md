# Third-party data and code

| What | Where | License | Used for |
|---|---|---|---|
| 20 Newsgroups (SetFit mirror) | https://huggingface.co/datasets/SetFit/20_newsgroups | see dataset card | `newsgroups` task |
| deepset/prompt-injections | https://huggingface.co/datasets/deepset/prompt-injections | Apache-2.0 | `injection` task |
| MASSIVE intents (mteb mirror) | https://huggingface.co/datasets/mteb/amazon_massive_scenario | CC-BY-4.0 | `massive_route` task (18-way utterance routing) |
| CLINC150 / clinc_oos | https://huggingface.co/datasets/clinc_oos | CC-BY-3.0 | `clinc_escalate` task (a `noul` on whether an utterance is out of scope) |
| banking77 (mteb parquet mirror) | https://huggingface.co/datasets/mteb/banking77 | CC-BY-4.0 | `banking20` task |
| LocalLLaMA/typed-decisions | https://huggingface.co/datasets/LocalLLaMA/typed-decisions | Apache-2.0 | `bench/tasks/typed_decisions.py`: the Laya / Jev-mode tables, the shipped typed heads, `demo/jev_mode.py`. Its gold is one teacher model's soft label per decision, not a human judgment, so accuracy on it is agreement with that teacher |
| Laya checkpoints (NandhaKishorM/laya) | https://github.com/NandhaKishorM/laya | Apache-2.0 | `bench/providers/laya.py`, run through their own `predict` API on the same decisions |
| NanoJev (TianyuCodings/NanoJev) | https://github.com/TianyuCodings/NanoJev | MIT | `bench/providers/nanojev_maze.py`, `nanojev_native_maze.py`: their frozen maze harness, baseline script and episode data |
| MLX and mlx-lm | https://github.com/ml-explore/mlx · https://github.com/ml-explore/mlx-lm | MIT | Optional `mlx` extra and `anyjev/backends/mlx.py` |

Datasets are downloaded at run time, never vendored. `bench/tasks/typed_paraphrases.json` (the rewordings used
by `bench.paraphrase_study` and the demo) was written here and carries the repo license.

# Methods implemented

- Contextual calibration: Zhao et al., ICML 2021, arXiv:2102.09690
- Batch calibration: Zhou et al., ICLR 2024, arXiv:2309.17249
- Permutation debiasing: Zheng et al., ICLR 2024, arXiv:2309.03882
- Temperature scaling: Guo et al., ICML 2017, arXiv:1706.04599
- L2 heads: shrunk linear discriminant analysis and dual-form ridge regression, textbook closed forms (`anyjev/heads.py`)
