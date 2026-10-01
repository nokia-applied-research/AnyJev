"""MLX-LM backend for direct next-token log-probability readout."""
from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np


class MLXBackend:
    """Thin ``mlx-lm`` backend supporting raw, L0 and L1 decisions."""

    def __init__(self, model_name: str, *, batch_size: int = 8,
                 trust_remote_code: bool = False, **load_kwargs: Any):
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ImportError("MLXBackend requires the 'mlx' extra: pip install anyjev[mlx]") from exc
        self.name = model_name
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        tokenizer_config = dict(load_kwargs.pop("tokenizer_config", {}) or {})
        self.model, self.tokenizer = load(
            model_name,
            tokenizer_config=tokenizer_config,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )

    def _pad_id(self) -> int:
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(self.tokenizer, attr, None)
            if value is not None:
                return int(value)
        return 0

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        import mlx.core as mx
        if len(prompts) != len(token_ids):
            raise ValueError("prompts and token_ids must have the same length")
        encoded = [list(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        if any(not row for row in encoded):
            raise ValueError("MLXBackend does not support empty prompts")
        results: List[np.ndarray] = []
        pad_id = self._pad_id()
        for start in range(0, len(encoded), self.batch_size):
            rows = encoded[start:start + self.batch_size]
            wanted = token_ids[start:start + self.batch_size]
            lengths = np.asarray([len(row) for row in rows], dtype=np.int32)
            width = int(lengths.max())
            batch = [row + [pad_id] * (width - len(row)) for row in rows]
            logits = self.model(mx.array(batch))
            if hasattr(logits, "logits"):
                logits = logits.logits
            last = logits[mx.arange(len(rows)), mx.array(lengths - 1), :].astype(mx.float32)
            logp = last - mx.logsumexp(last, axis=-1, keepdims=True)
            mx.eval(logp)
            for row, ids in zip(logp, wanted):
                if not ids:
                    results.append(np.empty(0, dtype=np.float64))
                else:
                    results.append(np.asarray(row[mx.array(list(ids))], dtype=np.float64))
        return results
