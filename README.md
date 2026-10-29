<div align="center">

<img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/banner.png" width="100%" alt="AnyJev: turn any LLM into a Jev-style decision model. Typed decisions, real probabilities, training-free or self-distilled. Qwen3-8B on BANKING77-20, raw readout to L0 with no labels: order-flip rate 0.230 to 0.073, accuracy 0.747 to 0.803, auto-decidable at 5% risk 7.7% to 46.3%.">

[![PyPI](https://img.shields.io/pypi/v/anyjev?color=3b82f6)](https://pypi.org/project/anyjev/)
[![Python](https://img.shields.io/pypi/pyversions/anyjev)](https://pypi.org/project/anyjev/)
[![CI](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml/badge.svg)](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/nokia-applied-research/AnyJev/blob/main/LICENSE)
[![Technical Report](https://img.shields.io/badge/Technical%20Report-PDF-b31b1b?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2610.00831)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Tacit-models-yellow)](https://huggingface.co/collections/morriszjm/tacit-6ac41d0b50af9e5417c5c234)
[![Tacitus](https://img.shields.io/badge/%F0%9F%A4%97%20Tacitus-31B-f97316)](https://huggingface.co/morriszjm/AnyJev-Tacitus-31B)
[![Decision Index](https://img.shields.io/badge/Decision%20Index%200.3-60.22-8b5cf6)](https://huggingface.co/spaces/multimodalart/jev-decision-index)
[![Decision Index 0.4.1](https://img.shields.io/badge/Decision%20Index%200.4.1-Tacitus%2066.47-6d28d9)](https://huggingface.co/spaces/multimodalart/jev-decision-index)

**English** · [简体中文](https://github.com/nokia-applied-research/AnyJev/blob/main/README.zh-CN.md) · [⚡ Serve it](#-serve-it) · [📊 Results](#-tacit-results) · [🏆 Leaderboard](#-anyjev-tacitus-on-the-decision-index) · [🧭 Roadmap](#-roadmap) · [📖 Levels](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/levels.md)

</div>

<p align="center">
  <b>Jiamu Zhang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Tianze Yang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Yucheng Shi</b><sup>2</sup> &nbsp;&nbsp;&nbsp; <b>Evan Chen</b><sup>1</sup>
  <br>
  <b>Zixiang Nie</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Kelly Wan</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Liangjie Hong</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Ninghao Liu</b><sup>3</sup> &nbsp;&nbsp;&nbsp; <b>Liang Wu</b><sup>1</sup>
</p>
<p align="center">
  <sub><sup>1</sup>&nbsp;Nokia, Sunnyvale, CA, USA &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>&nbsp;Tencent Hunyuan &nbsp;&nbsp;&nbsp;&nbsp; <sup>3</sup>&nbsp;The Hong Kong Polytechnic University</sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/flip.gif" width="100%" alt="Reverse the option order: the raw logit readout flips its answer, AnyJev L0 gives the same answer both ways">
  <br>
  <sub>Qwen3-8B on a real BANKING77 item. Every number is a model output.</sub>
</p>

> [!TIP]
> **🆕 Tacit: AnyJev's self-distilled models, 1.7B to 9B, are on
> [Hugging Face](https://huggingface.co/collections/morriszjm/tacit-6ac41d0b50af9e5417c5c234).** One
> forward pass per decision. With `adaptive=True`, a capped share of low-confidence decisions goes to
> the model's own reasoning. They run on transformers, in-process vLLM or a stock `vllm serve`.
> `pip install "anyjev[hf]"`, then `Tacit.from_pretrained("morriszjm/Tacit-9B")`.
> [Start here ↓](#-serve-it)

> [!NOTE]
> **🏆 AnyJev Tacitus on the Decision Index 0.4.1.** [AnyJev Tacitus 31B](https://huggingface.co/morriszjm/AnyJev-Tacitus-31B), fine-tuned from
> Gemma-4-31B and read with AnyJev, has a Full score of **66.47**, tied 3rd on the board on 2026-10-21.
> [Details ↓](#-anyjev-tacitus-on-the-decision-index)

> [!NOTE]
> **🏆 No training, on a public leaderboard.** The training-free `Decider` (L0, two rotations) on the
> unchanged Gemma-4-31B has a Full score of **60.22** on the [Decision Index 0.3](https://huggingface.co/spaces/multimodalart/jev-decision-index),
> 5th of 149 entries on 2026-10-12. [Details ↓](#-training-free-on-the-decision-index)

## ⚡ Serve it

**A Tacit model, in Python.**

```python
from anyjev import Tacit

tacit = Tacit.from_pretrained("morriszjm/Tacit-9B")          # transformers, one GPU
d = tacit.decide(state="Customer: my package was due last Monday and it still has not arrived.",
                 question="What does the customer want?",
                 options=["track_order", "cancel_order", "refund", "change_address"])
d["answer"], d["probs"], d["route"]    # an option, {option: probability}, "one_forward" or "cot"
```

`kind` is `"choice"` (default), `"yes_no"` or `"score"` (`options` are ordered levels, lowest
first, numbered from 1; `first_level=0` numbers them from 0). `decide_batch([...])` takes a list of such dicts.

**Send the hardest decisions to reasoning, with a cap.**

```python
tacit = Tacit.from_pretrained("morriszjm/Tacit-9B", adaptive=True, tau=0.5, max_cot_share=0.2, cot_window=1000)
```

A decision whose top two options are less than `tau` apart in log-probability is answered again with
thinking on. After the reasoning, the answer is read as a label distribution, so it also has
probabilities and cannot fail to parse. At most `max_cot_share` of the last `cot_window` decisions
escalate, however hard the traffic gets. Every escalation is reported (`route == "cot"`, with the
first pass kept), and `tacit.stats` counts them.

**On vLLM.** A stock server; the client needs only the tokenizer.

```bash
pip install "anyjev[client]"
vllm serve morriszjm/Tacit-9B --host 127.0.0.1 --port 8000
```

```python
tacit = Tacit.from_pretrained("morriszjm/Tacit-9B", engine="server",
                              base_url="http://127.0.0.1:8000", adaptive=True)
```

`engine="vllm"` (`pip install "anyjev[vllm]"`) runs vLLM in the same process instead.

**An HTTP endpoint for many clients sharing one cap.**

```bash
python -m anyjev.serve --model morriszjm/Tacit-9B --upstream http://127.0.0.1:8000 --adaptive --port 8100
curl -s 127.0.0.1:8100/v1/decide -H 'Content-Type: application/json' \
  -d '{"state": "Customer: my package has not arrived.", "question": "What does the customer want?", "options": ["track_order", "refund"]}'
```

`POST /v1/decide` takes one decision or `{"items": [...]}`; `GET /v1/stats` reports the escalation
share. The gateway listens on 127.0.0.1 and has no authentication of its own.

| argument | default | meaning |
|---|---|---|
| `adaptive` | `False` | send low-confidence decisions to the model's own reasoning |
| `tau` | `0.5` | low confidence: the log-probability gap between the top two options is below `tau` |
| `max_cot_share` | `0.2` | at most this share of the last `cot_window` decisions escalates; `None` removes the cap |
| `cot_window` | `1000` | how many recent decisions the cap counts; `None` counts every decision since loading |
| `cot_max_tokens` | `8192` | reasoning budget of one decision |
| `engine` | `"transformers"` | `"vllm"` runs vLLM in this process; `"server"` uses a running `vllm serve` at `base_url` |

**Any other model, training-free.** The `Decider` reads the same typed decisions from any open
causal LM, with no training and no labels.

```python
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend          # or VLLMBackend / SGLangBackend(url, model), LlamaCppBackend(gguf),
                                                  # StrataBackend(url, model, tokenizer_name=...)

d = Decider(HFBackend("Qwen/Qwen3-8B"))           # level="L0" by default
route = Question.choice("Which team should handle this?",
                        ["billing", "technical", "sales", "other"], name="route")
r = d.decide("My card was charged twice for one order.", [route])["route"]
r.distribution, r.level                           # {option: probability}, "L0"
```

L0 reads a K-option choice once per rotation of the options. With the **rotation budget**, it reads
only as many rotations as the decision needs:

```python
d = Decider(backend, adaptive_shifts=True, canonical_order=True)
d.calibrate_adaptive(route, unlabelled_tickets, target=0.01)   # a few hundred states, no labels
```

The stopping threshold is calibrated against our own full-cycle answer, so it needs no labels. At a
certified 1% disagreement it read 7.2 rotations instead of 18. That was 2.2× the decisions per second
on vLLM and 2.3×–2.7× on transformers, with accuracy unchanged
([docs/rotation_budget.md](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/rotation_budget.md)).

## ✨ What it is

Give the model a state and a **typed question** (a choice, a yes/no or a score). You get back a
**decision with a probability over the options**, read from the model's next-token distribution.
Nothing is parsed. There are two ways in:

- **AnyJev Training-Free (`Decider`)** works on any open LLM, with no training. Raw label logits
  change their answer when the options are reordered and carry the model's preference for some
  labels. L0 removes both without a single label.
- **AnyJev Self-Distilled (`Tacit`)** uses the Tacit checkpoints. They are trained by
  self-distillation: the model learns to give, in one forward pass, the answers it reaches when it
  reasons. No human labels and no other model are involved. One prefill per decision; with
  `adaptive=True`, a capped share of decisions goes to reasoning.

<div align="center">

| | ⚪&nbsp;raw&nbsp;logits | 🔵&nbsp;**L0**<br><sub>zero labels</sub> |
|:--|:--:|:--:|
| Labels required | none | **none** |
| Answer flips when options are reversed | 0.230 | **0.073** |
| Accuracy | 0.747 | **0.803** |
| Calibration error (ECE) | 0.240 | **0.184** |
| **Auto-decidable at ≤5% error** | **7.7%** | **46.3%** |

<sub>Qwen3-8B, BANKING77 20-way, 300 test items · `bench/results_v01/2026-09-22/Qwen__Qwen3-8B.json`</sub>

</div>

The last row matters most. Accuracy moves six points, but the share of decisions confident enough to
automate at ≤5% error goes from **7.7% to 46.3%**, with no labels at all.

## 📊 Tacit results

<div align="center">

| model | base | JevBench public (231) | adaptive | bev-decision test (46,320) | adaptive |
|:--|:--|:--:|:--:|:--:|:--:|
| [Tacit-9B](https://huggingface.co/morriszjm/Tacit-9B) | Qwen3.5-9B | 0.823 | **0.887** <sub>15.2%</sub> | 0.727 | **0.755** <sub>18.1%</sub> |
| [Tacit-8B](https://huggingface.co/morriszjm/Tacit-8B) | Qwen3-8B | 0.736 | — | 0.663 | — |
| [Tacit-4B](https://huggingface.co/morriszjm/Tacit-4B) | Qwen3-4B | 0.723 | **0.758** <sub>14.3%</sub> | 0.663 | **0.704** <sub>17.8%</sub> |
| [Tacit-2B](https://huggingface.co/morriszjm/Tacit-2B) | Qwen3.5-2B | 0.671 | — | 0.626 | — |
| [Tacit-1.7B](https://huggingface.co/morriszjm/Tacit-1.7B) | Qwen3-1.7B | 0.619 | — | 0.598 | — |

<sub>Accuracy on every item of both sets ([JevBench](https://github.com/fstandhartinger/jevbench) public
set; [bev-decision](https://huggingface.co/datasets/avbiswas/bev-decision) test split), in the order they
are stored. First column: one forward pass per decision (`adaptive=False`). Adaptive: `adaptive=True` at
the defaults (`tau=0.5`, at most 20% of every 1,000 decisions), with the share of decisions sent to
reasoning in small print. Served by `vllm serve` (vLLM 0.17.1), run with `scripts/eval_tacit.py` · `bench/results_tacit/2026-10-06/`</sub>

</div>

On average, escalation adds 567 (JevBench) and 458 (bev-decision) generated tokens per decision for Tacit-9B,
and 305 and 187 for Tacit-4B. Results with escalation for Tacit-8B, Tacit-2B and Tacit-1.7B come next.

## 🏆 AnyJev Tacitus on the Decision Index

[AnyJev Tacitus 31B](https://huggingface.co/morriszjm/AnyJev-Tacitus-31B) is a decision model fine-tuned from
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it), with an inference technique designed together with
the training. It reads every decision from one prefill through AnyJev's readout, with no generated tokens. On the
[Decision Index 0.4.1](https://huggingface.co/spaces/multimodalart/jev-decision-index) board it is tied 3rd by Full score (2026-10-21);
from edition 0.4 the Full score counts the private parts only.

<div align="center">

| Full score | same skills, private <sub>60%</sub> | new domains, private <sub>40%</sub> | public <sub>not counted</sub> | latency median / mean / p95 |
|:--:|:--:|:--:|:--:|:--:|
| **66.47** | 62.48 | 56.93 | 65.52 | 95 / 265 / 1,000 ms |

<sub>The board's maintainers ran the two private parts and measured the latency (single process, one RTX PRO 6000).
Served by vLLM 0.29.0 with FP8 at load and the model's vLLM plugin, anyjev 0.3.0. Engine, plugin and run settings:
[MorrisZJ/anyjev-tacitus](https://github.com/MorrisZJ/anyjev-tacitus); results: [decision-index-results](https://huggingface.co/datasets/morriszjm/decision-index-results/tree/d82c491a187b0f015b589ba813084fb695d98b78/runs/anyjev-tacitus-31b-v1) ·
`bench/results_decision_index/2026-10-21/anyjev-tacitus-31b.json`</sub>

</div>

## 🏆 Training-free on the Decision Index

The `Decider` at L0 on the unchanged [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it):
no training, no labels, two rotations per decision. On the [Decision Index 0.3](https://huggingface.co/spaces/multimodalart/jev-decision-index) board it is
5th of 149 entries by Full score (2026-10-12).

<div align="center">

| Full score | public <sub>20%</sub> | same skills, private <sub>50%</sub> | new domains, private <sub>30%</sub> | latency median / mean / p95 |
|:--:|:--:|:--:|:--:|:--:|
| **60.22** | 59.79 | 59.13 | 52.98 | 98 / 260 / 1,080 ms |

<sub>The board's maintainers ran the two private parts and measured the latency (single process, one RTX PRO 6000).
Served by `vllm serve --quantization fp8` (vLLM 0.29.0) with anyjev 0.3.0. Engine and run settings:
[examples/decision_index](https://github.com/nokia-applied-research/AnyJev/tree/main/examples/decision_index) (submitted as
[anyjev-l0-gemma](https://github.com/MorrisZJ/anyjev-l0-gemma)); model card:
[AnyJev-L0-Gemma-4-31B](https://huggingface.co/morriszjm/AnyJev-L0-Gemma-4-31B) ·
`bench/results_decision_index/2026-10-12/anyjev-l0-gemma-4-31b.json`</sub>

</div>

## 🧠 How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/how_it_works.png" width="100%" alt="How one decision is read: ask a typed question, read it over every cyclic shift of the options, divide out the label prior estimated without labels, and return a decision that carries its level">
</p>

| | needs | does | does **not** |
|---|---|---|---|
| `raw` | nothing | one prompt, softmax restricted to the option labels | correct order or label bias |
| `L0` | nothing | averages position bias out over the K rotations, divides out the label prior | calibrate against labels |
| Tacit `one_forward` | a Tacit checkpoint | one prompt, read like `raw` from a model trained to answer in one pass | correct order bias |
| Tacit `cot` | `adaptive=True` | reasons first, then reads the label distribution after the thought | run beyond its cap |

Every `Decision` carries its `level`, and `require="L0"` makes downstream code refuse a weaker one.
Every Tacit decision carries its `route`.
[The contract in full →](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/levels.md)

<sub>Every number in this README is read from committed JSON. Not affiliated with TypeSafe AI or Jev.</sub>

## 🧭 Roadmap

- [x] `choice`, `noul` and `score` from one prefill; L0 with zero labels
- [x] The rotation budget: L0 at about 7 of 18 rotations, with a stopping threshold certified without labels
- [x] **Tacit-1.7B, 2B, 4B, 8B and 9B** on Hugging Face; `Tacit` in the library on transformers, vLLM and `vllm serve`, with capped escalation to reasoning and an HTTP gateway
- [x] The evaluation harness (`scripts/eval_tacit.py`); results with escalation for Tacit-9B and Tacit-4B
- [x] The training-free path on the public Decision Index 0.3 board ([L0 on Gemma-4-31B](#-training-free-on-the-decision-index))
- [x] **AnyJev Tacitus 31B** on the Decision Index 0.4.1 board, tied 3rd ([details](#-anyjev-tacitus-on-the-decision-index))
- [ ] 🚧 Results with escalation for Tacit-8B, Tacit-2B and Tacit-1.7B
- [ ] 🚧 Label-free early exit: read a decision from part of the model's depth, choosing the depth by agreement with the full model
- [ ] **Agent-loop evaluation**: the same decisions inside a real agent
- [x] SGLang backend (`anyjev.backends.sglang`), contributed by @shentonyan
- [x] Strata backend (`anyjev.backends.strata`) for Strata's `/v1/completions`, repeatable reads with Strata's canonical reads
- [x] llama.cpp backend for GGUF models (`anyjev.backends.llamacpp`), contributed by @Tusm11
- [ ] More log-prob backends (MLX, Ollama), span readout beyond 26 options

Dated plan and help-wanted files: [ROADMAP.md](https://github.com/nokia-applied-research/AnyJev/blob/main/ROADMAP.md). Known limitations: [docs/limitations.md](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/limitations.md).

## 🤝 Contributing and citation

Backends are one file each, and several are **help wanted** ([ROADMAP.md](https://github.com/nokia-applied-research/AnyJev/blob/main/ROADMAP.md), [CONTRIBUTING.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CONTRIBUTING.md)). Changes: [CHANGELOG.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CHANGELOG.md). Credits: [CREDITS.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CREDITS.md).

Technical Report:
```bibtex
@misc{zhang2026anyjevtechnicalreport,
      title={AnyJev Technical Report}, 
      author={Jiamu Zhang and Tianze Yang and Yucheng Shi and Evan Chen and Zixiang Nie and Kelly Wan and Liangjie Hong and Ninghao Liu and Liang Wu},
      year={2026},
      eprint={2610.00831},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.00831}, 
}
```

Apache-2.0, see [LICENSE](https://github.com/nokia-applied-research/AnyJev/blob/main/LICENSE). The Tacit models carry their base models' Apache-2.0 license. Datasets keep their own licenses, see [THIRD_PARTY.md](https://github.com/nokia-applied-research/AnyJev/blob/main/THIRD_PARTY.md).
