"""SGLang backend through the native ``/generate`` API.

The native endpoint can score arbitrary token ids without generating a token:
append one placeholder token to the prompt, ask for its input logprob, and use
``token_ids_logprob`` for the labels AnyJev requested.

    python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000
    Decider(SGLangBackend("http://localhost:30000", "Qwen/Qwen3-8B"))

This backend needs a SGLang server and does not expose L2 hidden states.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import urllib.request
from typing import List, Sequence

import numpy as np


class SGLangBackend:
    """Next-token log-probabilities from SGLang's native generation API."""

    def __init__(self, base_url: str, model: str, tokenizer_name: str | None = None,
                 api_key: str = "EMPTY", workers: int = 16, timeout: float = 120.0):
        from transformers import AutoTokenizer

        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key = api_key
        self.workers = workers
        self.timeout = timeout
        self.source = tokenizer_name or model
        self.tokenizer = AutoTokenizer.from_pretrained(self.source)

    def _one(self, prompt: str, ids: Sequence[int]) -> np.ndarray:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        placeholder = self.tokenizer.pad_token_id
        if placeholder is None:
            placeholder = self.tokenizer.eos_token_id
        if placeholder is None:
            placeholder = 0
        body = {
            "input_ids": prompt_ids + [placeholder],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": len(prompt_ids),
            "token_ids_logprob": list(ids),
        }
        req = urllib.request.Request(
            self.base_url + "/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            out = json.load(response)

        rows = out.get("input_token_ids_logprobs_val")
        if not rows:
            raise RuntimeError("SGLang did not return input token log-probabilities")
        values = rows[-1]
        if values is None or len(values) != len(ids):
            raise RuntimeError(
                "SGLang did not return log-probabilities for every requested token id: "
                f"requested {len(ids)}, received {0 if values is None else len(values)}")
        if any(value is None for value in values):
            raise RuntimeError("SGLang returned a missing log-probability for a requested token id")
        return np.asarray(values, dtype=np.float64)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        with cf.ThreadPoolExecutor(self.workers) as executor:
            return list(executor.map(self._one, prompts, token_ids))
