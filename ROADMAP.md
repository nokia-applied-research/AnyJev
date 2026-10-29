# Roadmap

This file is the plan; `CHANGELOG.md` is what landed. A box is ticked only when the code, its tests,
the result JSON and the doc line are all on `main`. There are no dates here: the only target is the
version at the top of each section. Items marked **help wanted** are one-file PRs; the list at the
bottom says which file.

AnyJev has two directions. **Training-free**: read typed decisions from any open LLM (`Decider`,
levels `raw` and `L0`). **Self-distilled**: the Tacit models, which answer in one forward pass and can
send a capped share of decisions to their own reasoning (`Tacit`). `docs/levels.md` is the contract.

## Shipped

The version in brackets is the one an item first shipped in.

- [x] **L0, zero-label debiasing** (0.0.2). Cyclic-shift marginalisation over option positions plus a
  label-prior correction (batch prior by default, content-free opt-in); every `Decision` carries its
  level.
- [x] **Backends** (0.0.2). transformers (`HFBackend`), a vLLM OpenAI-compatible server
  (`VLLMBackend`, with a parity script), and a synthetic biased `FakeBackend` for CPU tests.
- [x] **Shared-prefix scoring** (0.1). The state is computed once and the K option layouts are scored
  against its KV cache.
- [x] **Enforceable levels** (0.1). `decide(..., require="L0")` raises `LevelError` below the asked level.
- [x] **Batch prior at strength 0.75 by default** (0.1), chosen by an offline replay of every prior
  rule on 230 (model, question) units.
- [x] **The rotation budget** (0.2). Stop reading rotations when the log-odds margin clears a threshold
  that `Decider.calibrate_adaptive` certifies against the full-K answer with a Clopper-Pearson bound,
  no labels; rotations turn a canonical listing so re-listing the options cannot change the answer.
  7.2 rotations of 18, 2.2x-2.7x the decisions per second; [docs/rotation_budget.md](docs/rotation_budget.md).
  Opt-in until the L0 tables are regenerated at the new defaults.
- [x] **Tacit models on Hugging Face** (0.3). Tacit-1.7B, 2B, 4B, 8B and 9B, one forward per decision,
  in the [Tacit collection](https://huggingface.co/collections/morriszjm/tacit-6ac41d0b50af9e5417c5c234);
  accuracy on JevBench (public, 231) and bev-decision (test, 46,320) in `bench/results_tacit/2026-10-06/`.
- [x] **`Tacit` in the library** (0.3). `anyjev.Tacit` on transformers, in-process vLLM and a running
  `vllm serve`; `adaptive=True` sends decisions whose top-two margin is under `tau` to the model's own
  reasoning and reads the answer as a label distribution afterwards; at most `max_cot_share` of the
  last `cot_window` decisions escalate. `python -m anyjev.serve` puts an HTTP gateway with one shared
  cap in front of a vLLM server. Tested on CPU with a fake engine (`tests/test_tacit.py`).
- [x] **The evaluation harness, and results with escalation** (unreleased). `scripts/eval_tacit.py` decides
  every item of the JevBench public set and the bev-decision test split through `Tacit` against `vllm serve`,
  one forward or adaptive, and writes the JSON the README tables are read from (`bench/results_tacit/2026-10-06/`):
  one forward for all five Tacit models, adaptive for Tacit-9B and Tacit-4B.
- [x] **llama.cpp backend** (unreleased). `LlamaCppBackend` runs GGUF models through llama-cpp-python at raw /
  L0, reading the last position's full-vocabulary log-probabilities; parity against `HFBackend` in
  `scripts/llamacpp_parity.py`. Contributed by @Tusm11 (#1, #2).
- [x] **SGLang backend** (unreleased). `SGLangBackend` reads the label log-probabilities from SGLang's
  native `/generate` endpoint without sampling; parity against `HFBackend` in `scripts/sglang_parity.py`,
  checked on SGLang 0.5.10. Contributed by @shentonyan (#12).
- [x] **Strata backend** (unreleased). `StrataBackend` reads the label log-probabilities from Strata's
  `/v1/completions` with `allowed_token_ids`, without sampling; checked against the same server's top-20 in
  `scripts/strata_parity.py` on Strata 0.1.53 (needs Strata 0.2.7 or later with Niko1221/Strata#3332 and #4056;
  repeatable reads with #4604).
- [x] **Releases on PyPI** (0.1.0, 0.2.0). Each was installed from the index into a fresh venv, with
  `__version__` matching `pyproject.toml`.

Removed in 0.3.0: the label-trained levels (`L1` temperature scaling, `L2` closed-form heads, routing,
test-time adaptation, `observe`), the shipped heads, the demos and the benchmark code. They remain in
`anyjev==0.2.0`.

## Now

- [ ] **Results with escalation for Tacit-8B, Tacit-2B and Tacit-1.7B.** Why: the READMEs show escalation
  for Tacit-9B and Tacit-4B only. *Done:* `scripts/eval_tacit.py --adaptive` on both sets for each, the JSON
  in `bench/results_tacit/`, and the empty cells filled.
- [ ] **Label-free early exit** (0.3.1). Why: a decision often does not need the model's full depth. A
  map into the final layer's basis, fitted without labels, reads the decision from a truncated
  forward, and the depth is chosen by agreement with the full model's answer. The hidden-state
  plumbing (`HFBackend.hidden_states_to`, `anyjev.truncate`, the vLLM pooler path) already ships.
  *Done:* `Decider` reads a truncated forward through the map, a `FakeBackend` test covers it, and the
  agreement-vs-depth JSON is committed.

## Next

- [ ] **Real agentic evaluation.** Why: every number so far is a decision in isolation. One public agent
  benchmark with the loop's decisions answered by the same LLM generating them, by L0, and by Tacit;
  task success, per-decision accuracy, and cost per decision and per episode. *Done:* one command,
  results JSON committed, the README quotes it whichever way it comes out.
- [ ] **Span readout for more than 26 options.** Why: letter labels cap `choice` at 26. Score each
  option string teacher-forced under the prompt (`sequence_logprobs` on the backend protocol).
  *Done:* a 77-way BANKING77 row at raw / L0 with flip and ECE, and a `FakeBackend` test with a planted
  length bias.
- [ ] **Conformal abstention.** Why: a probability still needs a rule for "do not answer".
  `Decider(target_error=...)` sets `decision.abstained`. *Done:* a `FakeBackend` test shows the realised
  error on answered items at or under the target on held-out synthetic data.
- [ ] **More log-prob backends: MLX, Ollama.** Why: the contract is one method,
  `next_token_logprobs(prompts, token_ids)`, and each engine is a file. *Done:* a parity script like
  `scripts/vllm_parity.py` agrees with transformers on argmax, and an `@pytest.mark.engine` smoke test
  exists. **help wanted.**

## Later

- [ ] **Multimodal states.** Vision backends (`AutoModelForImageTextToText`), one image task with ground
  truth; raw / L0 first.
- [ ] **Quantization.** Accuracy and ECE of L0 and of a Tacit model at bf16 / FP8 / W4, from JSON.
- [ ] **Label-free skew estimate for the batch prior.** The batch prior costs accuracy when the true
  label marginal is skewed, and no label-free rule tells that case from a biased model today.

## Help wanted

Each of these is one file, reviewed as one PR. Open an issue first if you want the slot.

- [ ] `anyjev/backends/mlx.py`, `ollama.py`: implement
  `next_token_logprobs`, add a parity script against `HFBackend`, mark the engine test
  `@pytest.mark.engine`.

## Non-goals

No base-model pretraining, no prompt-compilation framework, no routing logic beyond a decision's own
escalation. Those belong upstream or downstream of a decision layer.

## How items move

An item moves from Next to Now when someone is working on it, and from Now to Shipped when its
definition of done is met on `main`. A result row is a committed JSON plus the command that produced
it, with hardware and versions; if a run did not happen, the cell is empty. `CONTRIBUTING.md` has the
file contracts.
