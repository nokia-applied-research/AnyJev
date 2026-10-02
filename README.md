<div align="center">

<img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/banner.png" width="100%" alt="AnyJev — turn any LLM into a Jev-style decision model. Typed decisions, real probabilities, no fine-tuning. Order-flip rate 0.230 to 0.073 with zero labels; calibration error 0.240 to 0.095 and auto-decidable at 5% risk 7.7% to 52.0% with 100 to 500 labels.">

[![PyPI](https://img.shields.io/pypi/v/anyjev?color=3b82f6)](https://pypi.org/project/anyjev/)
[![Python](https://img.shields.io/pypi/pyversions/anyjev)](https://pypi.org/project/anyjev/)
[![CI](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml/badge.svg)](https://github.com/nokia-applied-research/AnyJev/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/nokia-applied-research/AnyJev/blob/main/LICENSE)

**English** · [简体中文](https://github.com/nokia-applied-research/AnyJev/blob/main/README.zh-CN.md) · [⚡ Serve it](#-serve-it) · [📊 Results](#-with-labels-l2) · [🧭 Roadmap](#-roadmap) · [📖 Levels](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/levels.md)

</div>

<p align="center">
  <b>Jiamu Zhang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Tianze Yang</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Yucheng Shi</b><sup>2</sup> &nbsp;&nbsp;&nbsp; <b>Liang Wu</b><sup>1</sup>
</p>
<p align="center">
  <sub><sup>1</sup>&nbsp;Nokia, Sunnyvale, CA &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>&nbsp;Tencent Hunyuan</sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/flip.gif" width="100%" alt="Reverse the option order: the raw logit readout flips its answer, AnyJev L0 gives the same answer both ways">
  <br>
  <sub>Qwen3-8B on a real BANKING77 item. Every number is a model output.</sub>
</p>

> [!TIP]
> **🆕 vLLM serves every level, L2 included.** An embed server's pooler hands back the hidden
> state a closed-form head reads, so a decision endpoint is a pooling server plus a few
> kilobytes of head. `python -m anyjev.pipeline <model>` converts, serves and measures in one
> command. [Start here ↓](#-serve-it)


## ⚡ Serve it

Three commands take a model off the Hub and put a calibrated decision endpoint in front of it.

```bash
pip install "anyjev[hf]"

# 1. keep the blocks a decision needs — usually about two thirds
python -m anyjev.truncate Qwen/Qwen2.5-7B-Instruct 18 ./qwen-b18

# 2. serve it. L2 reads a hidden state, so the pooler hands one back untouched
vllm serve ./qwen-b18 --task embed \
  --override-pooler-config '{"pooling_type":"LAST","normalize":false,"softmax":false}'
```

```python
from anyjev import Decider, Question
from anyjev.backends.vllm import VLLMBackend

d = Decider(VLLMBackend("http://localhost:8000", "./qwen-b18"), level="L2")
route = Question.choice("Which team should handle this?",
                        ["billing", "technical", "sales", "other"], name="route")

d.fit_head(route, states, labels, layers=[-1])   # 100–300 labels, one closed-form solve
d.decide(ticket, [route])["route"].distribution  # {"billing": 0.81, "technical": 0.07, ...}
```

**Without labels, turn on the rotation budget — recommended for any K-option `choice`.** L0 asks the
model once per option rotation so that no option is favoured by its position. Most decisions do not need
all K: read them one at a time, stop when the leader is far enough ahead, and the threshold can be
calibrated so the answer matches the full cycle's answer a stated fraction of the time — measured against
**our own full-strength readout, so it needs no labels at all.**

```python
d = Decider(VLLMBackend("http://localhost:8000", "./qwen-b18"), adaptive_shifts=True)
d.calibrate_adaptive(route, unlabelled_tickets, target=0.01)   # a few hundred states, no labels
d.decide_batch(tickets, route)      # diagnostics: shifts_used, stop_threshold
```

7.2 rotations instead of 18 at a certified 1% disagreement rate, **2.2× the decisions per second on
vLLM and 2.3×–2.7× on the transformers backend**, accuracy unchanged
([docs/rotation_budget.md](docs/rotation_budget.md)). It is opt-in in 0.2.0 only because the tables in
`docs/` were measured before it existed; it becomes the default when they are regenerated.

**An L2 deployment is a pooling server plus a few kilobytes of head.** No logits, no parsing, no
patched engine, and nothing generated. raw / L0 / L1 run the same way against a `--task generate`
server. A head fit through `transformers` and served by vLLM answers the same as one fit and
served on either alone — 99.0% identical answers, mean |dp| 0.0011 on BANKING77-20.

**Measure it on your own box instead of trusting ours:**

```bash
python -m anyjev.pipeline Qwen/Qwen2.5-7B-Instruct --labels-from banking20
```

That truncates, serves, fits a head, measures accuracy, ECE and ms per decision on held-out
states, shuts the server down, and repeats at full depth so there is something to compare
against. Timings are a median over `--repeats` passes with the spread printed next to them,
because on a shared machine a single pass can report the same configuration as both faster and
slower than the baseline.

> **Depth is usually a gain, not a trade.** Cutting Qwen2.5-7B from 28 blocks to 18 left accuracy
> slightly *higher* and calibration better, and was faster: a middle block is a better feature
> space for a linear head than the last one, where the remaining blocks are busy turning the
> answer into tokens. `--quantization fp8` is available and not recommended — it buys
> single-question latency and costs accuracy.

## ✨ What it is

Ask any open LLM a **typed question** — a choice, a yes/no, a score — and get a **decision with a
probability you can threshold**, read from one prefill of its next-token distribution. Nothing is
generated and nothing is parsed. Raw logits change their answer when you reorder the options and
their confidence cannot be trusted; AnyJev fixes the first with zero labels and the second with a
few hundred.

<div align="center">

| | ⚪&nbsp;raw&nbsp;logits | 🔵&nbsp;**L0**<br><sub>zero labels</sub> | 🟢&nbsp;**L1**<br><sub>+ temperature</sub> |
|:--|:--:|:--:|:--:|
| Labels required | none | **none** | 100–500 |
| Answer flips when options are reversed | 0.230 | **0.073** | 0.077 |
| Accuracy | 0.747 | **0.803** | 0.807 |
| Calibration error (ECE) | 0.240 | 0.184 | **0.095** |
| **Auto-decidable at ≤5% error** | **7.7%** | **46.3%** | **52.0%** |

<sub>Qwen3-8B, BANKING77 20-way, 300 test items · <a href="docs/results_bench.md">every ablation</a></sub>

</div>

The last row is the point: accuracy moves six points, but the share of traffic you can safely
automate goes **7.7% → 52.0%**. With raw logits a "0.9" is not trustworthy enough to act on, so
everything goes to a human. Once the probability means what it says, you can set a threshold.

## 📊 With labels: L2

A closed-form head per question, solved on 100–300 labels in seconds — no gradients, the model's
weights untouched — and read from one prompt stopped partway down.

<div align="center">

| model | L0, zero labels | **L2** | block | cost vs one forward |
|:--|:--:|:--:|:--:|:--:|
| Qwen3-1.7B | 0.494 | **0.730** | 18 / 28 | 0.70× |
| Qwen3-4B | 0.564 | **0.786** | 24 / 36 | 0.69× |
| Qwen3-8B | 0.647 | **0.771** | 24 / 36 | 0.68× |
| Qwen3-30B-A3B | 0.630 | **0.799** | 40 / 48 | — |
| Qwen3-32B | 0.700 | **0.798** | 52 / 64 | 0.84× |

<sub>LocalLLaMA/typed-decisions, 20 questions × 300 labels, 2,000 held-out decisions. Pooled ECE
0.03–0.05. Jev 0.727 and fine-tuned Laya 0.768 on the same set, as published by their authors ·
<a href="docs/results_exit.md">every cell</a></sub>

</div>

A 1.7B at 64% of its depth reaches the number Jev publishes; a 4B ties the fine-tuned 421M Laya.
**100 labels** already put the 8B head at 0.740. Heads for five Qwen3 models ship in
`anyjev-heads/`, 23 heads per model in one 1.8–4.4 MB file.

**A head maintains itself.** Only its feature mean and scale move afterwards, re-estimated from
**unlabelled** traffic, so it follows its question across rewordings and option orders on its own
— a reworded question drops the Qwen3-8B head from 0.77 to 0.65–0.70 and **30 unlabelled
requests** bring it back to 0.74–0.75, against 0.77 for a full relabelled refit. New labels are
needed only for a new question. [How the routing works →](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/method_v3.md)

## 🧠 The levels

<p align="center">
  <img src="https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/assets/how_it_works.png" width="100%" alt="How one decision is read: ask a typed question, read it over every cyclic shift of the options, divide out the label prior estimated without labels, and return a decision that carries its level">
</p>

| Level | Needs | Does | Does **not** |
|---|---|---|---|
| `raw` | nothing | restricted softmax over label tokens | anything about bias or calibration |
| `L0` | nothing | averages position bias out over the K rotations, divides out the label prior | calibrate the uncertainty |
| `L1` | 100–500 labels per question | temperature scaling on top of L0 | change the ranking |
| **`L2`** | **100–300 labels per question** | **a closed-form head on the hidden state partway down, one prompt per state** | **transfer to another question or model** |

Every `Decision` carries its `level`, and `require="L1"` makes downstream code refuse to act on a
weaker one. L0 costs K prefills for a K-option choice, or about 7 of 18 with the rotation budget on (`adaptive_shifts=True`, recommended — see [Serve it](#-serve-it)). **L2 costs less than one plain forward**.

`d.observe(q, state, label)` collects labels as they arrive and solves the head by itself at 30,
re-solving at 60, 120, … so day 0 runs at L0 with nothing and L2 arrives when the loop has fed it.
[The contract in full →](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/levels.md) · [the method →](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/method_v3.md)

**No GPU handy?** `python -m demo.jev_mode --backend fake` runs the whole thing on a synthetic
model in under a second.

<p align="center"><sub>
<a href="docs/jev_mode.md">Jev mode</a> ·
<a href="demo/games/README.md">2048 and Minesweeper</a> ·
<a href="docs/results_maze.md">NanoJev maze</a> ·
<a href="docs/when_l0_helps.md">when L0 helps</a> ·
<a href="docs/results_small_models.md">small models</a> ·
<a href="docs/research_log.md">research log, negative results included</a>
</sub></p>

<sub>Every number is regenerated from committed JSON (`bash scripts/regen_docs.sh`); a second run
from a clean checkout reproduced every zero-label number bit for bit. Not affiliated with TypeSafe
AI or Jev; rows published by their authors were not rerun here.</sub>

## 🧭 Roadmap

- [x] `choice`, `noul` and `score` from one prefill; L0 with zero labels; L1 artifacts
- [x] **L2**: a closed-form head per question, routing, label-free adaptation, `observe`
- [x] Shipped heads for five Qwen3 models; a packaged demo
- [ ] 🚧 **Speed optimization** *(ongoing)*: making every decision cheaper
- [x] **L2 on served engines**: vLLM, through an embed server's pooler or a truncated checkpoint (`anyjev.pipeline`, `anyjev.truncate`); SGLang not yet
- [ ] **Agent-loop evaluation**: the same decisions inside a real agent, against the LLM they replace
- [ ] Heads on the Hugging Face Hub, an interactive Space, a technical report
- [ ] More models (Llama, Gemma, Mistral, DeepSeek), span readout beyond 26 options, conformal abstention

Dated plan and help-wanted files: [ROADMAP.md](https://github.com/nokia-applied-research/AnyJev/blob/main/ROADMAP.md).

## 🔍 Limitations

- **On typed-decisions, "accuracy" is agreement with a teacher LLM.** The gold is the mean of three samples of one model; a fresh sample of that teacher agrees with it 0.735 of the time.
- **L2 is per question and per model.** Heads fit on other questions do not help a new one, and only Qwen3 heads ship. It needs hidden states, which transformers and a vLLM embed server both provide; other engines do not yet.
- **Calibration cannot fix a model that cannot answer.** On maze edges and Minesweeper no readout beats the trivial baseline.
- **L0 is not a free win everywhere.** The batch prior costs accuracy when one label dominates ([when L0 helps](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/when_l0_helps.md)).

<sub>Also: at most 26 options in the letter readout (a span readout is on the roadmap, not in the code); coverage at 5% risk is a high-variance estimate at n = 300; the headline tables are Qwen models; every decision here is scored in isolation, not inside an agent loop.</sub>

## 🤝 Contributing and citation

Backends and bench providers are one file each; several are **help wanted** ([ROADMAP.md](https://github.com/nokia-applied-research/AnyJev/blob/main/ROADMAP.md), [CONTRIBUTING.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CONTRIBUTING.md)). Changes: [CHANGELOG.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CHANGELOG.md). Credits: [CREDITS.md](https://github.com/nokia-applied-research/AnyJev/blob/main/CREDITS.md).

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

Software:
```bibtex
@software{anyjev2026,
  title  = {AnyJev: Turn any LLM into a Jev-style decision model},
  author = {Zhang, Jiamu and Yang, Tianze and Shi, Yucheng and Wu, Liang},
  year   = {2026},
  url    = {https://github.com/nokia-applied-research/AnyJev}
}
```

Apache-2.0, see [LICENSE](https://github.com/nokia-applied-research/AnyJev/blob/main/LICENSE). Datasets keep their own licenses, see [THIRD_PARTY.md](https://github.com/nokia-applied-research/AnyJev/blob/main/THIRD_PARTY.md).
