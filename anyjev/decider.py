"""Decider: the pipeline from (state, questions) to leveled decisions.

raw : one prompt, one permutation, restricted softmax. What every clone does.
L0  : permutation marginalization + label-free prior correction (batch mean
      by default, content-free probes optional). Zero labels.
L1  : L0, then a post-hoc calibrator fit on a labeled set for this question.
L2  : one prompt, the forward stopped at a fixed block, a closed-form head on that
      hidden state fit on a labeled set for this question (docs/jev_mode.md).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anyjev.calibrate.contextual import (
    DEFAULT_PROBES,
    apply_contextual,
    batch_prior,
    content_free_prior,
)
from anyjev.calibrate.permute import cyclic_shifts, flip_rate_across_perms, marginalize, spread_order
from anyjev.calibrate.posthoc import TemperatureScaler
from anyjev.calibrate.stopping import DEFAULT_LOG_MARGIN, choose_threshold, log_margin
from anyjev.heads import LinearHead, decode_array, encode_array
from anyjev.heads import fit_head as _fit_head
from anyjev.question import Question
from anyjev.readout import DEFAULT_SYSTEM, build_prompt, label_ids_for_perm, render_chat_parts, resolve_labels
from anyjev.result import LEVEL_ORDER, Decision, DecisionSet
from anyjev.state import render_state

LEVELS = ("raw", "L0", "L1", "L2", "auto")
PRIORS = ("batch", "content_free", "none")


def _softmax(lp: np.ndarray) -> np.ndarray:
    z = np.asarray(lp, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


#: L1 calibrators by the `method` prefix their artifacts carry. A calibrator registers here and
#: `load_artifact` finds it; anything unregistered raises instead of being read as a temperature.
#: Before 0.2.0 the dispatch fell through, so an artifact with any other method reached
#: `TemperatureScaler.from_dict` and died on a missing key -- or, worse, would have loaded as a
#: temperature of 1.0 and silently done nothing. The same shape of bug had already happened once with
#: head banks, which is why this is a table and not another `startswith`.
CALIBRATORS: Dict[str, Any] = {"temperature": TemperatureScaler}


class Decider:
    DEFAULT_PRIOR_STRENGTH = {"batch": 0.75, "content_free": 1.0, "none": 0.0}
    RANDOM_LISTING_MAX_K = 8            # fit_head(listing="auto"): random listing orders up to this many options

    def __init__(self, backend, *, level: str = "L0", prior: str = "batch", min_prior_n: int = 8,
                 prior_strength: Optional[float] = None,
                 max_permutations: Optional[int] = None, combine: str = "logmean",
                 cf_probes: Sequence[str] = DEFAULT_PROBES, record_content_free: bool = False,
                 system: str = DEFAULT_SYSTEM, shared_prefix="auto", shared_min_prefix_tokens: int = 256,
                 adaptive_shifts: bool = False, adaptive_min_shifts: int = 2,
                 adaptive_margin: Optional[float] = None, adaptive_stat: str = "logit",
                 adaptive_target: float = 0.01, adaptive_wave: int = 2,
                 adaptive_order: str = "spread", canonical_order: bool = False,
                 adapt="routed", adapt_min_n: int = 30):
        """prior_strength: exponent applied to the prior before dividing (1.0 = full correction,
        0.0 = none). Default 0.75 for the batch prior, 1.0 for the content-free prior: over 230
        (model, question) points the batch prior at 0.75 had the best mean gain and the smallest
        loss on questions whose true label marginal is skewed (docs/when_l0_helps.md).

        shared_prefix: "auto" (default) scores the permutations of one state through the
        backend's `score_shared` when it has one, a state has at least 3 permutations, and the
        shared prefix is at least `shared_min_prefix_tokens` long (below that, one batched
        forward over the full prompts is cheaper than a prefix forward plus a suffix forward);
        True shares whenever there are at least 2 permutations, regardless of length;
        False always sends full prompts.

        adaptive_shifts (opt-in in 0.2; the default once the shipped tables are
        regenerated under it): for choice questions with K >= 3, read the
        cyclic shifts in `spread_order` and stop once the running marginal is decided enough,
        after at least `adaptive_min_shifts`. The marginal is then an average over a subset of
        shifts, so the position bias is reduced rather than cancelled exactly -- which is why the
        threshold is chosen against a stated disagreement rate with the full-K answer instead of
        by hand. Set False to always read every shift.

        adaptive_stat: "logit" (default) compares the log-odds margin, top-1 minus top-2 of the
        marginal in log space; "prob" is the pre-0.6 rule (a probability gap plus unanimity across
        the shifts read) and is kept for callers who pinned `adaptive_margin`. The log-odds margin
        is the default because a probability gap saturates at 1 and stops discriminating exactly
        where the rule has to decide: over four (model, task) cells at K=18-20 it could not certify
        a 1% disagreement rate on two of them at any threshold (docs/rotation_budget.md).

        adaptive_target: the disagreement rate with the full-K answer that `calibrate_adaptive`
        certifies. Until a question is calibrated, the threshold is `DEFAULT_LOG_MARGIN` (8.5), the
        smallest value that certified 1% on all four of those cells at once, worth 1.9x-3.9x;
        calibrating per (model, question) was worth about 1.5x more.

        adaptive_margin: overrides the threshold for every question, in the units of
        `adaptive_stat`. None (default) means the calibrated value, or `DEFAULT_LOG_MARGIN`.

        canonical_order (opt-in in 0.2, and worth turning on whenever `adaptive_shifts`
        is): rotate a canonical listing of the options, ordered by their
        text, instead of the caller's listing. Reading every shift gives every option every position
        either way, but which options sit next to each other still follows the caller's order, and a
        partial shift budget does not cancel the position bias either -- so without this the decision
        can depend on the order the options were typed in. With it the prompts are a function of the
        option *set*, so listing the same options any other way returns the same probabilities
        exactly, at any shift budget. That is what makes `adaptive_shifts` safe to leave on. Note it
        is a different property from `diagnostics["order_flip_l0"]`, which reports whether the shifts
        read disagreed among themselves. Set False to reproduce pre-0.6 prompts.

        adaptive_wave: shifts requested per backend call (default 2). A wave overshoots the stop
        point slightly and in exchange waits on half as many rounds, and a round is a barrier the
        whole batch sits behind. Measured on Qwen2.5-7B / massive_route, 300 states, at a certified 1%
        target: on a vLLM server waves of 2 issue the same requests as waves of 1 (7.28 against 7.16)
        and run 1.7x faster in wall clock (2.22x over the full cycle against 1.32x); on the local
        transformers backend, where there are no round trips, 1 and 2 tie at 2.34x and wider waves
        lose. Hence 2: best remotely, tied-best locally. Wider is worse on both
        (`bench/results_layout/2026-09-27/{vllm,hf}_q25_massive.json`). `adaptive_min_shifts` is
        effectively rounded up to a multiple of the wave.

        adapt (L2): re-estimate a head's feature standardisation from the unlabelled states a
        question is asked on, once `adapt_min_n` have been seen (label-free test-time
        adaptation). "routed" (default): only for questions served by another layout's head (a
        rewording or a re-listing), where it recovers most of the lost accuracy; True: always,
        which costs about a point on the head's own layout; False: never. docs/jev_mode.md.

        level "auto": L2 where a head routes (`route`), else L1 where an artifact exists, else L0."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}")
        self.backend = backend
        self.level = level
        self.prior = prior
        self.min_prior_n = min_prior_n
        if prior_strength is not None and not 0.0 <= prior_strength <= 1.0:
            raise ValueError("prior_strength must be between 0 and 1")
        self.prior_strength = prior_strength
        self.max_permutations = max_permutations
        self.combine = combine
        self.cf_probes = tuple(cf_probes)
        self.record_content_free = record_content_free
        self.system = system
        if shared_prefix not in ("auto", True, False):
            raise ValueError("shared_prefix must be 'auto', True or False")
        self.shared_prefix = shared_prefix
        self.shared_min_prefix_tokens = shared_min_prefix_tokens
        if adaptive_min_shifts < 1:
            raise ValueError("adaptive_min_shifts must be >= 1")
        if adaptive_margin is not None and adaptive_margin < 0:
            raise ValueError("adaptive_margin must be >= 0")
        if adaptive_stat not in ("logit", "prob"):
            raise ValueError("adaptive_stat must be 'logit' or 'prob'")
        if not 0.0 < adaptive_target < 1.0:
            raise ValueError("adaptive_target must be between 0 and 1")
        if adaptive_wave < 1:
            raise ValueError("adaptive_wave must be >= 1")
        self.adaptive_shifts = adaptive_shifts
        self.adaptive_min_shifts = adaptive_min_shifts
        self.adaptive_margin = adaptive_margin
        self.adaptive_stat = adaptive_stat
        self.adaptive_target = adaptive_target
        self.adaptive_wave = adaptive_wave
        if adaptive_order not in ("spread", "consecutive"):
            raise ValueError("adaptive_order must be 'spread' or 'consecutive'")
        self.adaptive_order = adaptive_order
        self.canonical_order = canonical_order
        self.stats = {"backend_calls": 0, "flat_prompts": 0, "shared_groups": 0, "shared_prompts": 0,
                      "adaptive_items": 0, "adaptive_shifts_total": 0}
        self._stop: Dict[str, Dict[str, Any]] = {}      # q.key -> calibrated stopping certificate
        self._prefix_len_cache: Dict[str, int] = {}
        self._label_ids: Dict[tuple, List[int]] = {}
        self._artifacts: Dict[str, TemperatureScaler] = {}
        self._heads: Dict[str, Dict[str, Any]] = {}                # q.key -> {head, layer_abs, question_id, ...}
        if adapt not in ("routed", True, False):
            raise ValueError("adapt must be 'routed', True or False")
        self.adapt = adapt
        if adapt_min_n < 2:
            raise ValueError("adapt_min_n must be at least 2")
        self.adapt_min_n = int(adapt_min_n)
        self._adapt: Dict[str, Tuple[Any, Any, int]] = {}         # q.key -> (sum, sum of squares, n) of L2 features
        self._observed: Dict[str, Dict[str, Any]] = {}            # q.key -> labelled states seen through observe()
        self._running: Dict[str, Tuple[np.ndarray, int]] = {}   # q.key -> (sum p_pos_raw [P,K], n)
        self._cf_cache: Dict[str, np.ndarray] = {}                # q.key -> cf prior [P,K]
        # set when the backend's block loop (hidden_states_to) raised NotImplementedError and L2
        # features fell back to a full forward; L2 diagnostics then carry early_stop=False
        self.early_stop_error: Optional[str] = None

    # ---- public -------------------------------------------------------
    def decide(self, state: Any, questions: Sequence[Question], level: Optional[str] = None,
               require: Optional[str] = None) -> DecisionSet:
        """require: raise LevelError unless every result reaches this level. Use it where
        acting on an L0 probability would be a bug, e.g. `require="L1"` before thresholding."""
        level = level or self.level
        decs = self._run([state], list(questions), level)
        items = [decs[(0, qi)] for qi in range(len(questions))]
        if level == "auto":
            level = min((d.level for d in items), key=lambda lv: LEVEL_ORDER[lv], default="L0")
        out = DecisionSet(items, level)
        return out.require(require) if require else out

    def decide_batch(self, states: Sequence[Any], question: Question,
                     level: Optional[str] = None, require: Optional[str] = None) -> List[Decision]:
        """Many states, one question. The bench path, and the best path for
        batch prior estimation."""
        level = level or self.level
        decs = self._run(list(states), [question], level)
        out = [decs[(si, 0)] for si in range(len(states))]
        if require:
            for d in out:
                d.require(require)
        return out

    def calibrate(self, question: Question, states: Sequence[Any], labels: Sequence[int],
                  level: str = "L1") -> Dict[str, Any]:
        """Fit an L1 artifact (temperature over the L0 probabilities) for this question on labeled
        states; `level="L2"` fits a closed-form head instead (see `fit_head`). labels are option
        indices. Returns the artifact dict (store it; it is per model)."""
        if level == "L2":
            return self.fit_head(question, states, labels)
        if level != "L1":
            raise ValueError("calibrate() fits L1 (temperature) or L2 (closed-form head)")
        # The prior is estimated from the calibration set alone, in an isolated accumulator, so the
        # artifact is a pure function of (model, question, calibration set) and not of what this
        # decider happened to score earlier. The prior is frozen into the artifact and used at L1.
        saved = self._running.pop(question.key, None)
        try:
            decs = self.decide_batch(states, question, level="L0")
            calib_state = self._running.get(question.key)
        finally:
            if saved is not None:
                s, n = saved
                if calib_state is not None and calib_state[0].shape == s.shape:
                    self._running[question.key] = (s + calib_state[0], n + calib_state[1])
                else:
                    self._running[question.key] = saved
        probs = np.stack([d.probs for d in decs])
        scaler = TemperatureScaler.fit(probs, labels)
        used = decs[0].diagnostics.get("prior")
        scaler.prior = np.asarray(used, dtype=float) if used is not None else None
        scaler.prior_method = decs[0].diagnostics.get("prior_method", "none")
        scaler.prior_strength = float(decs[0].diagnostics.get("prior_strength", 0.0))
        scaler.n_calib = len(states)
        self._artifacts[question.key] = scaler
        return {"model": self.backend.name, "question": question.key, **scaler.to_dict()}

    def fit_head(self, question: Question, states: Sequence[Any], labels: Sequence[int], *,
                 layers: Optional[Sequence[int]] = None, kinds: Sequence[str] = ("lda", "ridge"),
                 n_folds: int = 5, seed: int = 0, listing: str = "auto") -> Dict[str, Any]:
        """L2: a closed-form head for this question on the model's hidden state at the end of the
        prompt, fit on labelled states (labels are option indices). No gradient step and no new
        model weights: one forward of the calibration states, then a shrunk-LDA / ridge solve; the
        block it reads and the head kind are chosen by out-of-fold NLL over the candidate blocks
        (`layers`, absolute block indices; default 50 / 60 / 70 / 85 / 100 % of depth). At
        decision time the forward stops at the chosen block, so the head also sets the cost.
        Needs a backend that exposes hidden states (local transformers) and at least max(8, 2K)
        labels; 100-300 per question in practice (docs/research_log.md, entry 5).

        listing: "random" shows each calibration state its options in a random order (labels
        stay option indices), so the head reads the answer independently of where it is listed
        and serves the question with its options in any order (flip under reversal 0.07 instead
        of 0.18 for a head fit on one order; entry 13); "canonical" keeps the question's order
        for every state; "auto" (default) is random up to 8 options and canonical above, because
        with 20 options and a few hundred states a random-listing head loses 6-18 points (the
        position code becomes noise the head cannot average out; entry 13). Returns the artifact
        dict (per model and question; store it, `load_artifact` restores it)."""
        if not (hasattr(self.backend, "hidden_states") or hasattr(self.backend, "hidden_states_to")):
            raise ValueError(f"backend {self.backend.name} exposes no hidden states; "
                             "L2 needs a local transformers backend")
        if listing not in ("auto", "random", "canonical"):
            raise ValueError("listing must be 'auto', 'random' or 'canonical'")
        if listing == "auto":
            listing = "random" if question.k <= self.RANDOM_LISTING_MAX_K else "canonical"
        n_blocks = int(self.backend.n_layers)
        if layers is None:
            candidates = sorted({max(1, int(round(f * n_blocks))) for f in (0.5, 0.6, 0.7, 0.85, 1.0)})
        else:
            candidates = sorted({(n_blocks + 1 + int(i)) if int(i) < 0 else int(i) for i in layers})
        y = np.asarray(labels, dtype=int)
        if len(y) != len(states):
            raise ValueError("one label per state")
        if len(y) and (y.min() < 0 or y.max() >= question.k):
            raise ValueError(f"labels must be option indices in [0, {question.k})")
        perms = None
        if listing == "random" and not question.ordered and question.k > 1:
            rng = np.random.RandomState(seed)
            perms = [list(rng.permutation(question.k)) for _ in states]
        feats = self._features(question, [render_state(s) for s in states], candidates, perms)
        best = None
        for kind in kinds:
            try:
                h = _fit_head(feats, y, question.k, kind=kind, n_folds=n_folds, seed=seed)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or h.cv["oof_nll"] < best.cv["oof_nll"]:
                best = h
        if best is None:
            raise ValueError(f"no head could be fit for {question.id}: need at least max(8, 2K) labels")
        best.params["listing"] = listing if perms is not None else "canonical"
        entry = self._entry(question, best, int(candidates[best.layer]))
        self._heads[question.key] = entry
        return self._head_artifact(question.key, entry)

    def observe(self, question: Question, state: Any, label: int, *, fit_at: int = 30,
                refit_factor: float = 2.0, **fit_kwargs) -> Optional[Dict[str, Any]]:
        """Record one labelled state for `question` and solve its L2 head once enough have arrived.

        The first head is solved when `fit_at` observations exist (never below max(8, 2K); on
        typed-decisions 20 labels already beat L1 and 100 pass Jev's published number), and re-solved
        each time the count grows by `refit_factor` (30, 60, 120, ...). Returns the artifact dict
        when a head was (re)solved on this call, else None. With `level="auto"` the question is
        answered at L0 until then and at L2 afterwards. The observations stay in memory
        (`observations(question)`) and go into `export_artifacts(include_observations=True)`, so
        a later refit, or a new base model, can reuse them. `fit_kwargs` go to `fit_head`."""
        y = int(label)
        if not 0 <= y < question.k:
            raise ValueError(f"label must be an option index in [0, {question.k})")
        rec = self._observed.setdefault(question.key, {"question": question, "states": [], "labels": [],
                                                         "next_fit": max(int(fit_at), 8, 2 * question.k), "fits": 0})
        rec["states"].append(state)
        rec["labels"].append(y)
        if len(rec["labels"]) < rec["next_fit"]:
            return None
        art = self.fit_head(question, rec["states"], rec["labels"], **fit_kwargs)
        rec["fits"] += 1
        rec["next_fit"] = max(rec["next_fit"] + 1, int(round(len(rec["labels"]) * float(refit_factor))))
        return art

    def observations(self, question: Question) -> Tuple[List[Any], List[int]]:
        """The (states, labels) recorded through `observe` for this question layout."""
        rec = self._observed.get(question.key)
        return (list(rec["states"]), list(rec["labels"])) if rec else ([], [])

    @staticmethod
    def _entry(question: Question, head: LinearHead, layer_abs: int) -> Dict[str, Any]:
        return {"head": head, "layer_abs": int(layer_abs), "question_id": question.id, "kind": question.kind,
                "text": question.text, "options": list(question.options)}

    def _head_artifact(self, key: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        head: LinearHead = entry["head"]
        return {"model": self.backend.name, "question": key, "question_id": entry["question_id"],
                "kind": entry.get("kind"), "text": entry.get("text"), "options": entry.get("options"),
                "layer_abs": int(entry["layer_abs"]), "n_blocks": int(getattr(self.backend, "n_layers", 0)),
                "hidden_size": int(head.W.shape[0]), **head.to_dict()}

    def _features(self, q: Question, state_texts: List[str], layers_abs: Sequence[int],
                  perms: Optional[Sequence[Sequence[int]]] = None) -> np.ndarray:
        """Hidden state at the last prompt position, [N, len(layers), d], under the canonical
        listing order (or `perms[i]` for state i). Through the block loop when the backend has
        one (the forward stops at the deepest requested block), else through a full forward.
        A block loop that raises NotImplementedError (transformers without `masking_utils`,
        i.e. < 4.53, or an architecture it does not cover, e.g. Gemma's scaled embeddings)
        falls back to the full forward: the same layers, the same head, only the early stop
        is lost; `early_stop_error` records why and L2 diagnostics report early_stop=False."""
        labels, _ = self._labels_for(q)
        tok = self.backend.tokenizer
        prompts = []
        for i, st in enumerate(state_texts):
            perm = list(perms[i]) if perms is not None else list(range(q.k))
            pre, suf = render_chat_parts(tok, build_prompt(st, q, perm, self.system, labels))
            prompts.append(pre + suf)
        layers = [int(x) for x in layers_abs]
        self.stats["backend_calls"] += 1
        self.stats["flat_prompts"] += len(prompts)
        if hasattr(self.backend, "hidden_states_to") and self.early_stop_error is None:
            try:
                feats = self.backend.hidden_states_to(prompts, layers, None, None, max_layer=max(layers))[0]
            except NotImplementedError as e:
                self.early_stop_error = str(e) or e.__class__.__name__
                feats = self.backend.hidden_states(prompts, layers)[0]
        else:
            feats = self.backend.hidden_states(prompts, layers)[0]
        return np.asarray(feats, dtype=np.float32)

    def route(self, q: Question) -> Optional[Tuple[Dict[str, Any], Optional[List[int]]]]:
        """The stored head that serves `q`, or None. Exact layout first; then the same kind and
        option texts under another wording (a reworded question routes to the head and is
        adapted label-free to its own wording, see `adapt`); then, for heads fit with
        listing="random", the same option set in another order (probabilities are mapped back by
        option text). Returns (entry, index map) with index map None when no reordering."""
        entry = self._heads.get(q.key)
        if entry is not None:
            return entry, None
        for e in self._heads.values():
            opts = e.get("options")
            if opts is None or e.get("kind") != q.kind or len(opts) != q.k:
                continue
            if tuple(opts) == tuple(q.options):
                return e, None
            if e["head"].params.get("listing") == "random" and set(opts) == set(q.options):
                return e, [opts.index(o) for o in q.options]
        return None

    def _adapted(self, key: str, head: LinearHead, feats: np.ndarray) -> Tuple[LinearHead, int, bool]:
        """Label-free test-time adaptation: the head's feature standardisation re-estimated from
        every state this question has been asked on (including this batch), once at least
        `adapt_min_n` have been seen. A rewording or a re-listing of a question moves its hidden
        states mostly by a shift and a rescaling, which this undoes without labels (entry 13:
        0.65-0.70 -> 0.74-0.75 with 30 unlabelled states on Qwen3-8B, refit with labels 0.77)."""
        s, ss, n = self._adapt.get(key, (0.0, 0.0, 0))
        f64 = feats.astype(np.float64)
        s, ss, n = s + f64.sum(0), ss + (f64 ** 2).sum(0), n + len(f64)
        self._adapt[key] = (s, ss, n)
        if n < self.adapt_min_n:
            return head, n, False
        mean = s / n
        scale = np.sqrt(np.clip(ss / n - mean ** 2, 0.0, None)) + 1e-6
        return dataclasses.replace(head, mean=mean, scale=scale), n, True

    def _run_heads(self, states: List[Any], questions: List[Question]) -> Dict[tuple, Decision]:
        """L2: one prompt per state, forward to the head's block, head probabilities."""
        state_texts = [render_state(s) for s in states]
        n_blocks = int(getattr(self.backend, "n_layers", 0))
        out: Dict[tuple, Decision] = {}
        for qi, q in enumerate(questions):
            routed = self.route(q)
            if routed is None:
                raise ValueError(f"no L2 head for question {q.id}; call fit_head() or load its artifact first")
            entry, index_map = routed
            head, layer = entry["head"], entry["layer_abs"]
            feats = self._features(q, state_texts, [layer])[:, 0]
            exact = entry is self._heads.get(q.key)
            adapted, n_seen = False, 0
            if self.adapt is True or (self.adapt == "routed" and not exact):
                head, n_seen, adapted = self._adapted(q.key, head, feats)
            probs = head.probs(feats)
            if index_map is not None:
                probs = probs[:, index_map]
            # blocks_executed is the head's block; early_stop says whether the forward really
            # stopped there (False after the block-loop fallback in _features: full forward)
            early_stop = hasattr(self.backend, "hidden_states_to") and self.early_stop_error is None
            for si in range(len(states)):
                diag = {"readout": f"head:{head.kind}", "exit_layer": layer, "blocks_executed": layer,
                        "n_blocks": n_blocks, "relative_depth": (layer / n_blocks) if n_blocks else None,
                        "early_stop": early_stop,
                        "temperature": head.temperature, "n_calib": head.n_calib,
                        "listing": head.params.get("listing", "canonical"),
                        "routed_from": None if exact else entry.get("question_id"),
                        "reordered": index_map is not None, "adapted": adapted, "adapt_n": n_seen,
                        "permutations": 1, "perms": [list(range(q.k))]}
                out[(si, qi)] = Decision(q, np.asarray(probs[si]), "L2", diag)
        return out

    def load_artifact(self, question: Question, artifact: Dict[str, Any]) -> None:
        if artifact.get("model") not in (None, self.backend.name):
            raise ValueError(f"artifact was fit on {artifact['model']}, backend is {self.backend.name}")
        if str(artifact.get("method", "")).startswith("head:"):
            # an L2 head reads the state under this question's exact layout (kind, K, wording)
            if artifact.get("question") not in (None, question.key):
                raise ValueError("head artifact was fit on a different question layout "
                                 f"({artifact['question']} != {question.key}); fit a head for this one")
            entry = self._entry(question, LinearHead.from_dict(artifact), int(artifact["layer_abs"]))
            entry["question_id"] = artifact.get("question_id", question.id)
            self._heads[question.key] = entry
            return
        if artifact.get("prior") is not None and artifact.get("question") not in (None, question.key):
            # the frozen prior is indexed by (permutation, position): it belongs to the option
            # order it was fit on. Temperature-only artifacts (0.0.2) still load anywhere.
            raise ValueError("artifact carries a prior fit on a different question layout "
                             f"({artifact['question']} != {question.key}); calibrate this layout")
        method = str(artifact.get("method", "temperature"))
        family = method.split(":", 1)[0] or "temperature"
        if family not in CALIBRATORS:
            raise ValueError(f"artifact method {method!r} has no loader; register the calibrator in "
                             f"anyjev.decider.CALIBRATORS (known: {', '.join(sorted(CALIBRATORS))})")
        self._artifacts[question.key] = CALIBRATORS[family].from_dict(artifact)

    def export_artifacts(self, include_observations: bool = False) -> Dict[str, Any]:
        """Every L1 artifact and L2 head this decider holds, keyed by question hash, plus the
        label-free adaptation statistics of every routed question (so a restart answers a
        reworded question as it did before) and, on request, the labelled states recorded by
        `observe`. JSON-serializable; `load_artifacts` restores all of it."""
        out: Dict[str, Any] = {"model": self.backend.name,
                               "artifacts": {k: {"model": self.backend.name, "question": k, **v.to_dict()}
                                             for k, v in self._artifacts.items()},
                               "heads": {k: self._head_artifact(k, e) for k, e in self._heads.items()}}
        adaptation = {k: {"sum": encode_array(np.asarray(s, dtype=np.float64)),
                          "sumsq": encode_array(np.asarray(ss, dtype=np.float64)), "n": int(n)}
                      for k, (s, ss, n) in self._adapt.items() if n > 0}
        if adaptation:
            out["adaptation"] = adaptation
        if include_observations and self._observed:
            out["observations"] = {k: {"question_id": r["question"].id, "kind": r["question"].kind,
                                       "text": r["question"].text, "options": list(r["question"].options),
                                       "states": list(r["states"]), "labels": list(r["labels"]),
                                       "fits": r["fits"], "next_fit": r["next_fit"]}
                                   for k, r in self._observed.items()}
        return out

    def save_artifacts(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.export_artifacts(), f, indent=1)

    def load_artifacts(self, path_or_dict) -> int:
        """Load artifacts saved by save_artifacts. Refuses artifacts fit on another model."""
        import json
        d = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict))
        if d.get("model") not in (None, self.backend.name):
            raise ValueError(f"artifacts were fit on {d['model']}, backend is {self.backend.name}")
        for key, art in d.get("artifacts", {}).items():
            self._artifacts[key] = TemperatureScaler.from_dict(art)
        for key, art in d.get("heads", {}).items():
            self._heads[key] = {"head": LinearHead.from_dict(art), "layer_abs": int(art["layer_abs"]),
                                "question_id": art.get("question_id", key), "kind": art.get("kind"),
                                "text": art.get("text"),
                                "options": list(art["options"]) if art.get("options") is not None else None}
        for key, st in d.get("adaptation", {}).items():
            self._adapt[key] = (decode_array(st["sum"]), decode_array(st["sumsq"]), int(st["n"]))
        for key, r in d.get("observations", {}).items():
            rec = self._observed.get(key)
            if rec is None:
                if not (r.get("kind") and r.get("options")):
                    continue
                q = Question(r["kind"], r["text"], tuple(r["options"]), r.get("question_id"),
                             ordered=(r["kind"] == "score"))
                rec = self._observed[key] = {"question": q, "states": [], "labels": [], "next_fit": 30, "fits": 0}
            rec["states"].extend(r.get("states", []))
            rec["labels"].extend(int(x) for x in r.get("labels", []))
            rec["fits"] = int(r.get("fits", rec["fits"]))
            rec["next_fit"] = int(r.get("next_fit", rec["next_fit"]))
        return len(d.get("artifacts", {})) + len(d.get("heads", {}))

    def strength(self) -> float:
        """The exponent applied to the prior in use (see prior_strength)."""
        if self.prior_strength is not None:
            return self.prior_strength
        return self.DEFAULT_PRIOR_STRENGTH[self.prior]

    def running_prior(self, question: Question) -> Optional[np.ndarray]:
        """The batch prior accumulated so far for this question, [P, K] by position, or None."""
        entry = self._running.get(question.key)
        if entry is None or entry[1] < self.min_prior_n:
            return None
        return batch_prior(entry[0][None] / entry[1])

    def reset_prior(self, question: Optional[Question] = None) -> None:
        """Forget the batch prior accumulated for `question`, or for every question if None.

        The prior keeps running across calls, so a long-lived Decider's L0 answers depend on
        everything it has scored before. After a reset the question starts where a fresh
        Decider does (no correction until `min_prior_n` items); heads, frozen L1 priors and
        the content-free cache are kept."""
        if question is None:
            self._running.clear()
        else:
            self._running.pop(question.key, None)

    # ---- internals ----------------------------------------------------
    def _labels_for(self, q: Question):
        """(labels, token ids) for this question kind and size, resolved once per tokenizer."""
        key = (q.kind, q.k)
        if key not in self._label_ids:
            self._label_ids[key] = resolve_labels(self.backend.tokenizer, q)
        return self._label_ids[key]

    def _score(self, prompts: List[str], prompt_ids: List[List[int]],
               prompt_parts: List[Tuple[str, str]]) -> List[np.ndarray]:
        """Prompts that share a prefix and the same label ids (the permutations of one
        state) go through the backend's score_shared in one group; the rest go flat."""
        out: List[Optional[np.ndarray]] = [None] * len(prompts)
        use_shared = self.shared_prefix is not False and hasattr(self.backend, "score_shared")
        groups: Dict[tuple, List[int]] = {}
        if use_shared:
            min_size = 2 if self.shared_prefix is True else 3
            for i, (pre, suf) in enumerate(prompt_parts):
                if suf:
                    groups.setdefault((pre, tuple(prompt_ids[i])), []).append(i)
            groups = {k: v for k, v in groups.items() if len(v) >= min_size}
            if self.shared_prefix == "auto":
                groups = {k: v for k, v in groups.items() if self._prefix_tokens(k[0]) >= self.shared_min_prefix_tokens}
        shared = {i for idxs in groups.values() for i in idxs}
        flat = [i for i in range(len(prompts)) if i not in shared]
        if flat:
            self.stats["backend_calls"] += 1
            self.stats["flat_prompts"] += len(flat)
            for i, lp in zip(flat, self.backend.next_token_logprobs([prompts[i] for i in flat],
                                                                   [prompt_ids[i] for i in flat])):
                out[i] = lp
        if groups:
            keys = list(groups)
            g = [(k[0], [prompt_parts[i][1] for i in groups[k]]) for k in keys]
            ids = [list(k[1]) for k in keys]
            self.stats["backend_calls"] += 1
            self.stats["shared_groups"] += len(g)
            self.stats["shared_prompts"] += len(shared)
            for k, lps in zip(keys, self.backend.score_shared(g, ids)):
                for i, lp in zip(groups[k], lps):
                    out[i] = lp
        return out  # type: ignore[return-value]

    def _cf_prior(self, q: Question, perms: List[List[int]], labels, perm_ids) -> np.ndarray:
        """Content-free prior per permutation, [P, K] in position space, cached per question."""
        if q.key not in self._cf_cache:
            prompts, ids, parts = [], [], []
            for perm, pids in zip(perms, perm_ids):
                for probe in self.cf_probes:
                    spec = build_prompt(probe, q, perm, self.system, labels)
                    pre, suf = render_chat_parts(self.backend.tokenizer, spec)
                    prompts.append(pre + suf)
                    ids.append(pids)
                    parts.append((pre, suf))
            lps = self._score(prompts, ids, parts)
            C = len(self.cf_probes)
            self._cf_cache[q.key] = np.stack([
                content_free_prior(np.stack([_softmax(lps[pi * C + c]) for c in range(C)]))
                for pi in range(len(perms))])
        return self._cf_cache[q.key]

    def _stop_rule(self, q: Question) -> Tuple[float, str]:
        """(threshold, statistic) for this question: an explicit `adaptive_margin` wins, then a
        certificate from `calibrate_adaptive`, then the measured default."""
        if self.adaptive_margin is not None:
            return float(self.adaptive_margin), self.adaptive_stat
        cert = self._stop.get(q.key)
        if cert is not None and cert.get("threshold") is not None:
            return float(cert["threshold"]), cert.get("stat", "logit")
        if self.adaptive_stat == "prob":
            return 0.1, "prob"          # the pre-0.6 default; measured at a 2.3% disagreement rate
        return DEFAULT_LOG_MARGIN, "logit"

    def calibrate_adaptive(self, question: Question, states: Sequence[Any], *,
                           target: Optional[float] = None, delta: float = 0.05,
                           level: str = "L0") -> Dict[str, Any]:
        """Certify a stopping threshold for one question from **unlabelled** states.

        Reads every cyclic shift of every state once through the ordinary pipeline -- so whatever
        prior and combine rule this Decider is configured with are included -- and then picks the
        cheapest threshold whose disagreement with the full-K answer is under `target` by a
        Clopper-Pearson upper bound at confidence 1 - `delta`. The reference is our own full-strength
        readout, never a label, which is what makes the guarantee free.

        Returns the certificate: the threshold, how many shifts it would have read on these states,
        and the bound that was achieved. A question with no certificate falls back to
        `DEFAULT_LOG_MARGIN`. If nothing can be certified at this target the threshold is None and
        every shift is read, which is the safe direction.

        Spend a few hundred states on it; at a 1% target, 300 states allow three disagreements and
        the bound is what stops that from being fitted too tightly.
        """
        if question.kind != "choice" or question.ordered or question.k < 3:
            raise ValueError("adaptive shifts only apply to unordered choice questions with k >= 3")
        if not states:
            raise ValueError("calibrate_adaptive needs states")
        target = self.adaptive_target if target is None else target
        want_cf = level != "raw" and (self.prior == "content_free" or self.record_content_free)
        record: List[List[Tuple[float, int]]] = []
        self._run_adaptive_choice([render_state(s) for s in states], question, level, want_cf,
                                  record=record)
        margins = np.asarray([[m for m, _ in tr] for tr in record], dtype=np.float64)
        winners = np.asarray([[w for _, w in tr] for tr in record], dtype=int)
        threshold, info = choose_threshold(margins, winners, target, self.adaptive_min_shifts, delta)
        cert = {"threshold": threshold, "stat": "logit", "question_id": question.id, **info}
        self._stop[question.key] = cert
        return cert

    def _run_adaptive_choice(self, state_texts: List[str], q: Question, level: str, want_cf: bool,
                             record: Optional[List[List[Tuple[float, int]]]] = None) -> List[Decision]:
        """Sequential cyclic shifts with an early stop per state. See __init__ for the rule.

        `record` is the calibration hook: given a list, every shift is read and the running
        (log-odds margin, winner) after each one is appended per state, so `calibrate_adaptive` can
        choose a threshold offline from the same pipeline that will serve."""
        tok = self.backend.tokenizer
        labels, ids = self._labels_for(q)
        perms = self._perms(q, level)
        perm_ids = [label_ids_for_perm(q, ids, perm) for perm in perms]
        P, K, n = len(perms), q.k, len(state_texts)
        cf_prior = self._cf_prior(q, perms, labels, perm_ids) if want_cf else None
        lp_rows: List[List[np.ndarray]] = [[] for _ in range(n)]        # per state, per shift read
        p_rows: List[List[np.ndarray]] = [[] for _ in range(n)]
        used: List[List[int]] = [[] for _ in range(n)]
        sums, counts = self._running.get(q.key, (np.zeros((P, K)), 0))
        if sums.shape != (P, K):
            sums, counts = np.zeros((P, K)), 0
        shift_sum = np.zeros((P, K))                                     # this call's per-shift totals
        shift_n = np.zeros(P, dtype=int)
        active = list(range(n))

        frozen = self._artifacts.get(q.key) if level == "L1" else None
        frozen_ok = frozen is not None and frozen.prior is not None and frozen.prior.shape == (P, K)

        def prior_for(sidx: int) -> Optional[np.ndarray]:
            if frozen_ok:
                return frozen.prior[sidx]
            if self.prior == "content_free":
                return cf_prior[sidx]
            if self.prior != "batch":
                return None
            tot = sums[sidx] + shift_sum[sidx]
            m = counts + shift_n[sidx]
            if m >= self.min_prior_n:
                return batch_prior((tot / m)[None])
            # later shifts see only the hard items: fall back to the pooled position profile
            pooled_n = counts * P + shift_n.sum()
            if pooled_n >= self.min_prior_n:
                return batch_prior(((sums.sum(0) + shift_sum.sum(0)) / pooled_n)[None])
            return None

        strength = frozen.prior_strength if frozen_ok else self.strength()

        def corrected(si: int) -> np.ndarray:
            rows = []
            for sidx, p in zip(used[si], p_rows[si]):
                pr = prior_for(sidx)
                rows.append(apply_contextual(p, np.power(pr, strength)) if pr is not None else p)
            return np.stack(rows)

        order = spread_order(P) if self.adaptive_order == "spread" else list(range(P))
        threshold, stat = self._stop_rule(q)
        wave = max(1, int(self.adaptive_wave))
        if record is not None:
            threshold, wave = None, 1             # calibration reads every shift, one at a time, and
            trace = [[] for _ in range(n)]        # decides offline once it has the whole trace

        def decided(si: int) -> bool:
            """The stopping rule. `logit` compares the log-odds margin, which does not saturate;
            `prob` keeps the pre-0.6 behaviour (a probability gap plus unanimity across the shifts
            read) for callers who pinned `adaptive_margin`."""
            pc = corrected(si)
            marg = marginalize(pc, [perms[sidx] for sidx in used[si]], self.combine)
            if record is not None:
                trace[si].append((log_margin(marg), int(np.argmax(marg))))
                return False
            if stat == "logit":
                return log_margin(marg) >= threshold
            winners = {perms[sidx][int(np.argmax(pc[j]))] for j, sidx in enumerate(used[si])}
            top = np.sort(marg)[::-1]
            return len(winners) == 1 and top[0] - top[1] >= threshold

        for start in range(0, P, wave):
            batch = order[start:start + wave]
            if start > 0 and (start >= self.adaptive_min_shifts or record is not None):
                active = [si for si in active if not decided(si)]
            if not active:
                break
            # one backend call per wave: `adaptive_wave > 1` trades a little precision in the stop
            # point for fewer round trips, which is what a remote engine charges for.
            prompts, pids, parts, who, which = [], [], [], [], []
            for r in batch:
                for si in active:
                    pre, suf = render_chat_parts(tok, build_prompt(state_texts[si], q, perms[r],
                                                                  self.system, labels))
                    prompts.append(pre + suf)
                    pids.append(perm_ids[r])
                    parts.append((pre, suf))
                    who.append(si)
                    which.append(r)
            for si, r, lp in zip(who, which, self._score(prompts, pids, parts)):
                p = _softmax(lp)
                lp_rows[si].append(lp)
                p_rows[si].append(p)
                used[si].append(r)
                shift_sum[r] += p
                shift_n[r] += 1
        if record is not None:
            for si in range(n):
                decided(si)                       # the margin after the final shift
                record.append(trace[si])
        # fold this call into the running prior (per shift, only the items that ran it)
        # stored as a P x K sum with a single count: use the shift-0 count, which every item ran
        scale = (shift_n[0] / np.maximum(shift_n, 1))[:, None]
        self._running[q.key] = (sums + shift_sum * scale, counts + int(shift_n[0]))

        out: List[Decision] = []
        for si in range(n):
            p_pos_raw = np.stack(p_rows[si])
            used_perms = [perms[sidx] for sidx in used[si]]
            pc = corrected(si)
            probs = marginalize(pc, used_perms, self.combine)
            raw_probs = marginalize(p_pos_raw[:1], used_perms[:1])
            achieved = "L0"
            diag: Dict[str, Any] = {
                "answer_mass": float(np.exp(np.stack(lp_rows[si])).sum(axis=1).mean()),
                "raw_probs": raw_probs, "permutations": len(used_perms), "perms": used_perms,
                "p_pos_raw": p_pos_raw, "shifts_used": len(used_perms), "adaptive": True,
                "stop_threshold": threshold, "stop_stat": stat, "stop_calibrated": q.key in self._stop,
                "prior_method": self.prior if prior_for(used[si][0]) is not None else "none",
                "prior_strength": strength if prior_for(used[si][0]) is not None else 0.0,
                "prior": (np.stack([prior_for(sidx) for sidx in used[si]])
                          if prior_for(used[si][0]) is not None else None),
                "cf_prior": cf_prior[used[si]] if cf_prior is not None else None,
                "batch_prior": None,
                "order_flip_raw": flip_rate_across_perms(p_pos_raw, used_perms),
                "order_flip_l0": flip_rate_across_perms(pc, used_perms),
                "l0_probs": probs,
            }
            if level == "L1":
                scaler = self._artifacts.get(q.key)
                if scaler is None:
                    raise ValueError(f"no L1 artifact for question {q.id}; call calibrate() first")
                probs = scaler.apply(probs)
                achieved = "L1"
                diag["temperature"] = scaler.temperature
            self.stats["adaptive_items"] += 1
            self.stats["adaptive_shifts_total"] += len(used_perms)
            out.append(Decision(q, np.asarray(probs), achieved, diag))
        return out

    def _prefix_tokens(self, prefix: str) -> int:
        n = self._prefix_len_cache.get(prefix)
        if n is None:
            n = len(self.backend.tokenizer.encode(prefix, add_special_tokens=False))
            if len(self._prefix_len_cache) > 4096:
                self._prefix_len_cache.clear()
            self._prefix_len_cache[prefix] = n
        return n

    def _perms(self, q: Question, level: str) -> List[List[int]]:
        if level == "raw" or q.ordered:
            return [list(range(q.k))]
        if q.kind == "noul":
            return [[0, 1], [1, 0]]
        shifts = cyclic_shifts(q.k, self.max_permutations)
        if not self.canonical_order:
            return shifts
        # Rotate a canonical listing rather than the caller's. Reading every shift already gives
        # every option every position, but *which options sit next to each other* still follows the
        # caller's order, and options attend to one another (research log entry 21), so the readout
        # is not otherwise invariant to how the list was typed -- and it is much less so when only
        # a few shifts are read. Ordering by the option text first makes the prompts a function of
        # the option *set*, so any two listings of the same options produce the same decision
        # exactly, at any shift budget. Equal texts fall back to the caller's order, which is
        # harmless because equal options are interchangeable.
        canon = sorted(range(q.k), key=lambda i: (str(q.options[i]), i))
        return [[canon[i] for i in perm] for perm in shifts]

    def _run(self, states: List[Any], questions: List[Question], level: str) -> Dict[tuple, Decision]:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if level == "auto":
            groups: Dict[str, List[int]] = {"L2": [], "L1": [], "L0": []}
            for qi, q in enumerate(questions):
                groups["L2" if self.route(q) else ("L1" if q.key in self._artifacts else "L0")].append(qi)
            out_auto: Dict[tuple, Decision] = {}
            for lv, idxs in groups.items():
                if idxs:
                    sub = self._run(states, [questions[i] for i in idxs], lv)
                    for (si, j), dec in sub.items():
                        out_auto[(si, idxs[j])] = dec
            return out_auto
        if level == "L2":
            return self._run_heads(states, questions)
        tok = self.backend.tokenizer
        state_texts = [render_state(s) for s in states]
        want_cf = level != "raw" and (self.prior == "content_free" or self.record_content_free)

        # adaptive choice questions take the sequential path; everything else the batched one
        if self.adaptive_shifts and level != "raw":
            adaptive = [qi for qi, q in enumerate(questions) if q.kind == "choice" and not q.ordered and q.k >= 3]
            if adaptive:
                out: Dict[tuple, Decision] = {}
                for qi in adaptive:
                    for si, dec in enumerate(self._run_adaptive_choice(state_texts, questions[qi], level, want_cf)):
                        out[(si, qi)] = dec
                rest = [qi for qi in range(len(questions)) if qi not in set(adaptive)]
                if rest:
                    sub = self._run(states, [questions[qi] for qi in rest], level)
                    for (si, j), dec in sub.items():
                        out[(si, rest[j])] = dec
                return out

        # 1. collect every prompt once. Content-free probes (shared across states) get their own
        #    registry and their own forward call, so the batch composition of the real prompts --
        #    and with it their bf16 logits -- never depends on whether the probes are cached yet.
        #    Without this, a fresh Decider and a used one score the same states slightly differently,
        #    and `record_content_free` (a diagnostics flag) would move the numbers.
        def registry():
            index: Dict[str, int] = {}
            ids_: List[List[int]] = []
            parts: List[Tuple[str, str]] = []

            def add(spec, ids: List[int]) -> int:
                pre, suf = render_chat_parts(tok, spec)
                text = pre + suf
                if text not in index:
                    index[text] = len(ids_)
                    ids_.append(ids)
                    parts.append((pre, suf))
                return index[text]

            return index, ids_, parts, add

        prompt_index, prompt_ids, prompt_parts, add = registry()
        cf_index, cf_ids, cf_parts, add_cf = registry()

        plan = {}
        for qi, q in enumerate(questions):
            labels, ids = self._labels_for(q)
            perms = self._perms(q, level)
            cf_rows = []
            perm_ids = [label_ids_for_perm(q, ids, perm) for perm in perms]
            if want_cf and q.key not in self._cf_cache:
                for perm, pids in zip(perms, perm_ids):
                    cf_rows.append([add_cf(build_prompt(probe, q, perm, self.system, labels), pids)
                                    for probe in self.cf_probes])
            for si, st in enumerate(state_texts):
                real_rows = [add(build_prompt(st, q, perm, self.system, labels), pids)
                             for perm, pids in zip(perms, perm_ids)]
                plan[(si, qi)] = (perms, real_rows, cf_rows)

        # 2. score every prompt: shared-prefix groups where the backend supports it, flat otherwise.
        #    Real states and content-free probes are scored in separate calls (see step 1).
        def ordered(index: Dict[str, int]) -> List[str]:
            prompts = [None] * len(index)
            for text, i in index.items():
                prompts[i] = text
            return prompts

        logprobs = self._score(ordered(prompt_index), prompt_ids, prompt_parts)
        cf_logprobs = self._score(ordered(cf_index), cf_ids, cf_parts) if cf_index else []

        # 3. per question: raw position-space distributions, priors
        out: Dict[tuple, Decision] = {}
        for qi, q in enumerate(questions):
            perms, _, cf_rows = plan[(0, qi)]
            p_pos_raw_all = []
            for si in range(len(states)):
                lp_real = np.stack([logprobs[r] for r in plan[(si, qi)][1]])   # [P, K]
                p_pos_raw_all.append((lp_real, np.stack([_softmax(lp) for lp in lp_real])))

            cf_prior = None
            if want_cf:
                if cf_rows:
                    self._cf_cache[q.key] = np.stack([
                        content_free_prior(np.stack([_softmax(cf_logprobs[r]) for r in rows]))
                        for rows in cf_rows])                                      # [P, K]
                cf_prior = self._cf_cache[q.key]
            prior_used = b_prior = None
            frozen = self._artifacts.get(q.key) if level == "L1" else None
            if level != "raw":
                stack = np.stack([p for _, p in p_pos_raw_all])                    # [N, P, K]
                s, n = self._running.get(q.key, (np.zeros(stack.shape[1:]), 0))
                self._running[q.key] = (s + stack.sum(axis=0), n + len(stack))
                b_prior = self.running_prior(q)
                if frozen is not None and frozen.prior is not None and frozen.prior.shape == stack.shape[1:]:
                    prior_used = frozen.prior          # L1: the prior the artifact was fit with
                elif self.prior == "batch":
                    prior_used = b_prior
                elif self.prior == "content_free":
                    prior_used = cf_prior

            # 4. assemble per state
            for si, (lp_real, p_pos_raw) in enumerate(p_pos_raw_all):
                answer_mass = float(np.exp(lp_real).sum(axis=1).mean())
                raw_probs = marginalize(p_pos_raw[:1], perms[:1])
                diag: Dict[str, Any] = {"answer_mass": answer_mass, "raw_probs": raw_probs,
                                        "permutations": len(perms), "perms": perms,
                                        "p_pos_raw": p_pos_raw}
                if level == "raw":
                    probs, achieved = raw_probs, "raw"
                else:
                    strength = (frozen.prior_strength if frozen is not None and frozen.prior is not None
                                else self.strength())
                    p_pos = (apply_contextual(p_pos_raw, np.power(prior_used, strength))
                             if prior_used is not None else p_pos_raw)
                    probs = marginalize(p_pos, perms, self.combine)
                    achieved = "L0"
                    diag.update({
                        "prior_method": (("frozen:" + frozen.prior_method)
                                         if frozen is not None and frozen.prior is not None
                                         else (self.prior if prior_used is not None else "none")),
                        "prior_strength": strength if prior_used is not None else 0.0,
                        "prior": prior_used,
                        "cf_prior": cf_prior,
                        "batch_prior": b_prior,
                        "order_flip_raw": flip_rate_across_perms(p_pos_raw, perms),
                        "order_flip_l0": flip_rate_across_perms(p_pos, perms),
                        "l0_probs": probs,
                    })
                    if level == "L1":
                        scaler = self._artifacts.get(q.key)
                        if scaler is None:
                            raise ValueError(f"no L1 artifact for question {q.id}; call calibrate() first")
                        probs = scaler.apply(probs)
                        achieved = "L1"
                        diag["temperature"] = scaler.temperature
                out[(si, qi)] = Decision(q, np.asarray(probs), achieved, diag)
        return out
