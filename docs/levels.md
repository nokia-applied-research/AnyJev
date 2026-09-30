# What each level does, and does not do

Every `Decision` carries a `level`: `raw`, `L0`, `L1` or `L2`. This page is the contract behind that
field. `auto` is a request you can pass to `decide()` (serve the best level available for each
question); it is never what a decision reports.

## raw

One prompt, options in the order you gave them, softmax over the label tokens
at the answer position. This is what every open Jev clone does, and it is
exactly `max_tokens=1` plus logprobs.

What you get: a ranking. What you do not get: a probability. Two biases are
baked in:

- **Prior bias.** The model prefers some labels regardless of the input
  ("Yes" over "No", "A" over "D", common words over rare ones).
- **Position bias.** The model prefers some positions in the option list.
  Reorder the options and the argmax can change.

`raw` exists so the bench can show the gap. Do not ship it.

## L0: debiased, zero labels

Two training-free corrections, on by default.

**Prior correction.** Estimate the model's prior over the labels without
labels, divide the real distribution by it, renormalize. Two estimators:

- *Batch calibration* (Zhou et al., 2024), the default, applied at strength 0.75 (the prior is
  raised to `prior_strength` before dividing; 1.0 is the full correction). The prior is the mean
  predicted distribution over real inputs, kept running per question across
  calls and used once it has seen `min_prior_n` items (default 8). Before
  that, no prior correction is applied and `diagnostics["prior_method"]`
  says `none`. `Decider.reset_prior(question)` (or `reset_prior()` for all
  questions) starts the running prior over, e.g. between two measured runs
  through one long-lived Decider. In the bench it was the low-variance choice: +1 to +2
  accuracy points and a large ECE improvement on every model and task, with
  no task where it hurt by more than a point. It assumes the label marginal
  of the batch is not extreme.
- *Contextual calibration* (Zhao et al., 2021), opt-in via
  `prior="content_free"`. The prior is the answer distribution on
  content-free inputs (`N/A`, empty, `[MASK]`), computed once per question
  and cached, in a forward call of their own so the probes never change the
  batch composition (and with it the bf16 logits) of the real states. High variance: +7 to +12 points
  over raw on the prompt-injection `noul` on the three headline models
  (`L0-perm+cf` rows in `bench/results_v01/2026-09-22/`), but it loses points on ordinal `score`
  questions and on one model's `noul` questions in the typed-decisions set. For some questions the
  model's answer to an empty input is an honest answer, not a label prior, and
  dividing by it pushes every real answer the other way. The bench prints
  both (`L0-perm+bc` and `L0-perm+cf`), so measure before switching.

When each correction helps and when the prior hurts, measured over 230 (model, question) points:
[when_l0_helps.md](when_l0_helps.md). Short version: permutation is the safe half, the batch prior
hurts on questions whose true label marginal is skewed.

**Cyclic-shift marginalization** (Zheng et al., 2024). For a `choice` with K
options, show the list in K rotations so every option sits at every position
once, and combine. AnyJev combines in log space (geometric mean) by default:
if the position bias is additive in logit space, this removes it exactly and
the result no longer depends on how you listed the options. Arithmetic-mean
combination, the form in the paper, is available as `combine="mean"`.

For `noul`, the two phrasings "Answer Yes or No" and "Answer No or Yes" are
both read and combined. For `score`, the bins are ordinal and are never
permuted; only the prior correction applies.

The shifts can rotate a **canonical** listing of the options, ordered by their text
(`canonical_order=True`; opt-in in 0.2). Reading every shift gives every option
every position either way, but which options sit next to each other still follows
the caller's order and the options attend to one another, so without this the
decision can depend on how the list was typed. With it the prompts are a function
of the option *set*: the same options listed any other way return identical
probabilities, at any shift budget.

Cost: at most K prompts per choice question (2 for noul, 1 for score), all sharing
the state prefix. `max_permutations` caps K. With the rotation budget on, not all K are read:
`adaptive_shifts=True` reads them in `spread_order`, two per round
(`adaptive_wave`), and stops when the running marginal's log-odds margin clears a
threshold. The threshold is not a taste setting -- `Decider.calibrate_adaptive(q,
states, target=0.01)` reads every shift of a batch of **unlabelled** states once
and returns the cheapest threshold whose disagreement with the full-K answer is
under `target` by a Clopper-Pearson upper bound, so the guarantee is "the decision
the full cycle would have made, 1 - target of the time" and it costs no labels.
Until a question is calibrated the threshold is `DEFAULT_LOG_MARGIN`. Measured:
7.2 shifts of 18 at a certified 1% target on Qwen2.5-7B / massive_route, 2.2x on vLLM and 2.3x-2.7x
on transformers
([rotation_budget.md](rotation_budget.md)). `adaptive_shifts=False` reads every shift.

What L0 does not do: it does not make the model's own uncertainty
calibrated. A model that is overconfident on everything is still overconfident
after L0. That needs labels.

## L1: calibrated on labels

Temperature scaling fit on 100 to 500 labeled examples of the same question,
applied on top of L0. The prior used to score the calibration set is estimated from that set
alone and frozen into the artifact, so the artifact is a pure function of (model, question,
calibration set) and L1 decisions do not depend on what else the decider has scored (an
independent reproduction found the earlier running prior made L1 drift by up to 0.007 in
coverage at 5% risk). Two consequences worth knowing. The prior needs no labels but it does
need states: with 200 calibration items on a 20-way question it is a noisier estimate than the
batch prior L0 accumulates over everything it has seen, which on the smallest models costs up to
two points of AURC against the earlier, history-dependent L1 while ECE still improves; a larger
calibration set tightens it. And the prior is indexed by (permutation, position), so it belongs
to the option order it was fit on: `load_artifact` refuses an artifact whose prior was fit on a
different layout of the same question. The fitted temperature and the prior form a small JSON
artifact keyed by (model, question hash). Loading an artifact fit on a different model is an
error.

What L1 does not do: survive distribution shift beyond the calibration set,
or fix a model that is simply wrong. It reshapes confidence; it does not
change the ranking.

Planned for L1: Dirichlet calibration for multi-class, histogram binning,
and split-conformal abstention with a user-set target error rate.

## L2: a closed-form head on the hidden state

`Decider.fit_head(question, states, labels)` or `calibrate(..., level="L2")`. One forward of the
labelled states, then a shrunk-LDA or ridge solve on the model's hidden state at the end of the
prompt (the same state the raw readout projects with `lm_head`), the block it reads and the
temperature chosen by out-of-fold NLL on the calibration set. No gradient step, no new model
weights: a `[hidden, K]` matrix, a bias, a standardisation vector and a temperature per (model,
question), solved in seconds on a CPU.

What you get: the most accurate level on the typed-decisions questions (K <= 5), on banking20 and on
injection; on newsgroups the shipped 1.7B / 4B / 32B heads (0.585 / 0.630 / 0.735 on their 200
validation items) sit at or below L0 (0.610 / 0.667 / 0.737 on the bench's 300). On LocalLLaMA/typed-decisions with 300 labels per question the head reaches 0.771 on
Qwen3-8B, 0.786 on Qwen3-4B and 0.730 on Qwen3-1.7B where L1 is 0.648 / 0.567 / 0.499
(`docs/results_exit.md`, from `bench/results_exit/2026-09-22/<model>.jevmode.json`); with 100 labels
the 8B head is at 0.740 (`Qwen__Qwen3-8B.labels.json`). Pooled ECE is 0.03-0.05, the same range as L1.

What it costs: less per decision than every other level. The head reads a block at about two thirds
of the model's depth, so the forward stops there (`diagnostics["blocks_executed"]`); on Qwen3-8B that
is 0.68x the batched time of one plain forward (`Qwen__Qwen3-8B.latency.json`), and there is one
prompt per state instead of K cyclic shifts.

What it does not do: transfer to another question. A head is fit for one question on one model
and a head fit on other questions does not help a new one (question-agnostic heads scored at or
below L0 on every held-out question, `bench/results_universal/2026-09-22/`). The same question
reworded or with its options re-listed does route to its head (`Decider.route`), and a label-free
re-estimation of the head's feature mean and scale on the states it is asked on recovers most of
the accuracy a rewording costs (`Decider(adapt="routed")`; `docs/jev_mode.md`). Labels can arrive one at a
time: `Decider.observe(question, state, label)` records them and solves the head itself at 30 (then 60, 120,
...), so with `level="auto"` a question moves from L0 to L2 on its own, and the adaptation statistics and the
observations round-trip through `export_artifacts` / `load_artifacts`. A model's set of heads ships as
`anyjev-heads/<model>.json` with the arrays stored as base64 float32 (exact; 1.8-4.4 MB per model;
plain number lists load too). It needs a backend
that exposes hidden states (local transformers; backends that expose only log-probabilities stop at L1), at least
max(8, 2K) labels and in practice 100-300. Heads with up to 8 options are fit on random listing
orders, so their flip under reversal is 0.07; wider option lists keep
the canonical order.

## auto: a request, not a level

`decide(state, questions, level="auto")` serves each question at the best level available: L2 where
a head routes (exact layout, or the same options under another wording or order), else L1 where a
temperature artifact exists, else L0. The decision that comes back carries the level it was
actually served at, so `require="L2"` still raises on a question that fell back.

## Reading the bench columns

- `acc`, `macro_f1`: are the answers right.
- `brier`, `nll`: are the probabilities right (lower is better).
- `ece`: expected calibration error on top-1 confidence, 15 equal-mass bins.
- `flip`: fraction of items whose argmax changes when the option list is
  reversed (`choice`) or the phrasing is swapped between "Yes or No" and
  "No or Yes" (`noul`). Measured at every level. This is the number the
  clones' READMEs admit to and nobody tabulates.
- Ablation rows, all from the same forward passes: `L0-cf` content-free
  prior alone, `L0-bc` batch prior alone, `L0-perm` permutation alone,
  `L0-perm+cf` permutation plus content-free prior, `L0-perm+bc`
  permutation plus batch prior, `L0` whatever prior the run was configured
  with (default: batch, so `L0` equals `L0-perm+bc`).
- `cov@5%`: fraction of items you can answer, ordered by confidence, before
  the error rate on the answered set exceeds 5 percent. Higher is better.
