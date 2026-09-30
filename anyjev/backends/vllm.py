"""vLLM backend over the OpenAI-compatible server.

Two server shapes, because vLLM gives one task per instance, and between them they
cover every level.

**raw / L0 / L1** -- a generate server. One request per prompt with `max_tokens=1`,
`allowed_token_ids` restricted to the label tokens and `logprobs=K`. Start it with
`--logprobs-mode processed_logprobs`: vLLM's default, `raw_logprobs`, computes logprobs
before `allowed_token_ids` masks the vocabulary, so the top K can hold other tokens in
place of labels. With processed logprobs the K entries are exactly the labels. A label
missing from them raises `LabelTokenError`. Prefix caching makes the K permutations of
one state cheap.

    vllm serve Qwen/Qwen3-8B --enable-prefix-caching --logprobs-mode processed_logprobs
    Decider(VLLMBackend("http://localhost:8000", "Qwen/Qwen3-8B"))

**L2** -- an embed server whose pooler is told to return the last position's hidden
state untouched. That vector is what a closed-form head reads, and the head itself
runs on the CPU in microseconds, so an L2 deployment needs no logprobs at all: a
pooling server plus a few kilobytes of weights.

    vllm serve Qwen/Qwen3-8B --task embed --enable-prefix-caching \
      --override-pooler-config '{"pooling_type":"LAST","normalize":false,"softmax":false}'
    Decider(VLLMBackend("http://localhost:8000", "Qwen/Qwen3-8B"), ...).fit_head(q, states, y,
                                                                                layers=[-1])

Measured against `transformers` on Qwen2.5-7B: cosine 0.9998 between the two vectors,
the difference being bf16 kernel choice. Only the **final** layer is available this way,
so the depth truncation of the local backend does not apply -- which costs almost nothing
in accuracy (Qwen3-8B typed-decisions: 0.767 at the last block against 0.771 at block 24)
and gives up the compute saving in exchange for continuous batching and prefix caching.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import urllib.request
from typing import List, Optional, Sequence

import numpy as np

from anyjev.readout import LabelTokenError


class VLLMBackend:
    def __init__(self, base_url: str, model: str, tokenizer_name: str | None = None,
                 api_key: str = "EMPTY", workers: int = 16, timeout: float = 120.0):
        from transformers import AutoTokenizer

        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key = api_key
        self.workers = workers
        self.timeout = timeout
        # `model` is what the server answers to, which may be a `--served-model-name` alias;
        # the tokenizer and the config have to come from something loadable.
        self.source = tokenizer_name or model
        self.tokenizer = AutoTokenizer.from_pretrained(self.source)
        self._n_layers: int | None = None

    def _one(self, prompt: str, ids: Sequence[int]) -> np.ndarray:
        body = {
            "model": self.name, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
            "logprobs": len(ids), "allowed_token_ids": list(ids),
        }
        req = urllib.request.Request(
            self.base_url + "/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]   # {token_str: logprob}
        # map back by token id: decode each id the same way the server renders it
        by_id = {}
        for tid in ids:
            tok = self.tokenizer.decode([tid])
            conv = self.tokenizer.convert_ids_to_tokens(tid)
            for key in (tok, conv):
                if key in top:
                    by_id[tid] = float(top[key])
                    break
        missing = [tid for tid in ids if tid not in by_id]
        if missing:
            raise LabelTokenError(
                f"label token ids {missing} are not in the server's top_logprobs {sorted(top)}; "
                "start vLLM with --logprobs-mode processed_logprobs so that logprobs are taken "
                "after allowed_token_ids")
        return np.array([by_id[tid] for tid in ids], dtype=np.float64)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        with cf.ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, prompts, token_ids))

    # ---- L2: the last position's hidden state, from a pooling server ------------------
    @property
    def n_layers(self) -> int:
        """The model's real block count, read from its config.

        A pooling server can only return the final layer, but it must still be *numbered* the
        way the local backend numbers it, or an artifact stops being portable: a head fit
        locally records `layer_abs = 28` and a server that calls the same layer `1` refuses to
        load it, for no reason but arithmetic."""
        if self._n_layers is None:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(self.source)
            self._n_layers = int(getattr(cfg, "num_hidden_layers", 0))
        return self._n_layers

    def hidden_states(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None,
                      token_ids=None, positions=None):
        """(feats [N, 1, d], lps, None) from `/v1/embeddings`.

        `layers` may only name the single available layer (0, 1 or -1); anything else is an
        error rather than a silently substituted vector, because a head fit on block 24 and
        served on block 36 is a wrong answer with no symptom. `token_ids` is not served here:
        an embed server has no logits, and L2 does not need them."""
        final = self.n_layers
        if layers is not None and any(int(x) not in (-1, final) for x in layers):
            raise ValueError(f"a vLLM pooling server serves only the final layer ({final}); "
                             f"asked for {list(layers)}. Fit the head with layers=[-1].")
        if token_ids is not None:
            raise ValueError("an embed server returns no logits; use a generate server for "
                             "raw / L0 / L1 and this one for L2")
        if positions is not None:
            raise ValueError("a pooling server returns one vector per prompt, not per position")
        with cf.ThreadPoolExecutor(self.workers) as ex:
            vecs = list(ex.map(self._embed, prompts))
        feats = np.stack(vecs)[:, None, :].astype(np.float32)
        return feats, [None] * len(prompts), None

    def _embed(self, prompt: str) -> np.ndarray:
        body = {"model": self.name, "input": prompt, "encoding_format": "float"}
        req = urllib.request.Request(
            self.base_url + "/v1/embeddings", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        return np.asarray(out["data"][0]["embedding"], dtype=np.float32)
