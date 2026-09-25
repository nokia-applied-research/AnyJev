"""llama.cpp backend for GGUF models, through llama-cpp-python (L0 / L1).

One prompt at a time: clear the KV memory, decode the prompt with logits requested
for its final token only, take the full-vocabulary log-softmax of that one row and
read the requested ids. Nothing is sampled and no other row is materialized.

    pip install "anyjev[llamacpp]"
    Decider(LlamaCppBackend("qwen2.5-0.5b-instruct-f16.gguf"))
    Decider(LlamaCppBackend("qwen2.5-0.5b-instruct-f16.gguf", tokenizer="Qwen/Qwen2.5-0.5B-Instruct"))

Tokenizer. By default the GGUF's own vocabulary and chat template are used, so no
transformers install is needed. With `tokenizer` (a Hugging Face name or object) that
tokenizer renders the prompts and maps the labels, exactly as HFBackend does; then every
prompt is tokenized by both sides and must agree token for token, and every label id must
name the same text in both vocabularies. A mismatch raises TokenMismatchError instead of
silently reading the wrong logit. Parity against HFBackend: scripts/llamacpp_parity.py.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

MIN_LLAMA_CPP = (0, 3, 16)   # llama_memory_clear / llama_get_memory / llama_get_logits_ith(-1)


class TokenMismatchError(ValueError):
    """The Hugging Face tokenizer and the GGUF vocabulary disagree on a prompt or a label id."""


def _import_llama_cpp():
    try:
        import llama_cpp
    except ImportError as e:
        raise ImportError('LlamaCppBackend needs llama-cpp-python: pip install "anyjev[llamacpp]"') from e
    found = tuple(int(p) for p in llama_cpp.__version__.split(".")[:3] if p.isdigit())
    if found < MIN_LLAMA_CPP:
        need = ".".join(map(str, MIN_LLAMA_CPP))
        raise ImportError(f"LlamaCppBackend needs llama-cpp-python>={need}, found {llama_cpp.__version__}")
    return llama_cpp


class GGUFTokenizer:
    """The GGUF's own tokenizer behind the small interface anyjev.readout uses:
    `encode(text, add_special_tokens=False)`, `chat_template`, `apply_chat_template`."""

    def __init__(self, llm):
        self._llm = llm
        meta = getattr(llm, "metadata", None) or {}
        self.chat_template: Optional[str] = meta.get("tokenizer.chat_template") or None
        self.bos_token = self._piece(llm.token_bos())
        self.eos_token = self._piece(llm.token_eos())

    def _piece(self, tid: int) -> str:
        if tid is None or tid < 0:
            return ""
        return self._llm.detokenize([tid], special=True).decode("utf-8", errors="replace")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # special=True parses control tokens written in the text (<|im_start|> ...), as HF encode does
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=add_special_tokens, special=True))

    def decode(self, ids: Sequence[int]) -> str:
        return self._llm.detokenize(list(ids), special=True).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = True, **kwargs: Any):
        """Render the GGUF chat template the way transformers renders a chat template
        (sandboxed Jinja, trim_blocks / lstrip_blocks, `raise_exception`, `tojson`)."""
        if not self.chat_template:
            raise ValueError("this GGUF has no tokenizer.chat_template")
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise ValueError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = lambda x, indent=None, **_: json.dumps(x, ensure_ascii=False, indent=indent)
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        text = env.from_string(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token, eos_token=self.eos_token, **kwargs)
        return self.encode(text) if tokenize else text


class LlamaCppBackend:
    def __init__(self, model_path: str, tokenizer: Any = None, name: Optional[str] = None,
                 n_ctx: int = 4096, n_batch: int = 512, n_gpu_layers: int = 0,
                 verbose: bool = False, **llama_kwargs: Any):
        lib = _import_llama_cpp()
        self._lib = lib
        self.llm = lib.Llama(model_path=model_path, n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=n_gpu_layers,
                             logits_all=False, verbose=verbose, **llama_kwargs)
        self.name = name or os.path.splitext(os.path.basename(model_path))[0]
        self.n_ctx = int(lib.llama_n_ctx(self.llm.ctx))
        self.n_batch = int(lib.llama_n_batch(self.llm.ctx))
        self.n_vocab = int(self.llm.n_vocab())
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._hf = tokenizer                      # None: the GGUF vocabulary is the only one
        self.tokenizer = tokenizer if tokenizer is not None else GGUFTokenizer(self.llm)
        self._checked_ids: set = set()
        self._batch = lib.llama_batch_init(self.n_batch, 0, 1)

    # ---- token checks ------------------------------------------------------------
    def _prompt_tokens(self, prompt: str) -> List[int]:
        ids = list(self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
        if self._hf is not None:
            ref = list(self._hf.encode(prompt, add_special_tokens=False))
            if ids != ref:
                k = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref)))
                raise TokenMismatchError(
                    f"prompt tokenizes differently in the GGUF and in {type(self._hf).__name__} "
                    f"(lengths {len(ids)} vs {len(ref)}, first difference at token {k}: "
                    f"{ids[k:k + 3]} vs {ref[k:k + 3]}); the two vocabularies do not match")
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) > self.n_ctx:
            raise ValueError(f"prompt is {len(ids)} tokens, the context holds {self.n_ctx}; raise n_ctx")
        return ids

    def _check_label_ids(self, ids: Sequence[int]) -> None:
        for tid in ids:
            tid = int(tid)
            if tid in self._checked_ids:
                continue
            if not 0 <= tid < self.n_vocab:
                raise TokenMismatchError(f"label token id {tid} is outside the GGUF vocabulary ({self.n_vocab})")
            if self._hf is not None:
                ours = self.llm.detokenize([tid], special=True).decode("utf-8", errors="replace")
                theirs = self._hf.decode([tid])
                # surrounding whitespace is compared loosely: sentencepiece decoders drop a leading space
                if ours.strip() != theirs.strip():
                    raise TokenMismatchError(f"label token id {tid} is {ours!r} in the GGUF "
                                             f"and {theirs!r} in the Hugging Face tokenizer")
            self._checked_ids.add(tid)

    # ---- forward -----------------------------------------------------------------
    def _clear_memory(self) -> None:
        self._lib.llama_memory_clear(self._lib.llama_get_memory(self.llm.ctx), True)

    def _last_row_logits(self, tokens: List[int]) -> np.ndarray:
        """Decode `tokens` from an empty cache and return the final position's logits
        (float64 copy). Only that one position is flagged for output."""
        lib, ctx, b = self._lib, self.llm.ctx, self._batch
        self._clear_memory()
        n = len(tokens)
        for start in range(0, n, self.n_batch):
            chunk = tokens[start:start + self.n_batch]
            b.n_tokens = len(chunk)
            for j, tid in enumerate(chunk):
                b.token[j] = tid
                b.pos[j] = start + j
                b.n_seq_id[j] = 1
                b.seq_id[j][0] = 0
                b.logits[j] = 0
            if start + len(chunk) == n:
                b.logits[len(chunk) - 1] = 1
            rc = lib.llama_decode(ctx, b)
            if rc != 0:
                raise RuntimeError(f"llama_decode returned {rc} at tokens {start}..{start + len(chunk)}")
        row = lib.llama_get_logits_ith(ctx, -1)
        return np.ctypeslib.as_array(row, shape=(self.n_vocab,)).astype(np.float64)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for prompt, ids in zip(prompts, token_ids):
            self._check_label_ids(ids)
            logits = self._last_row_logits(self._prompt_tokens(prompt))
            m = logits.max()
            lse = m + np.log(np.exp(logits - m).sum())       # full-vocabulary log-softmax
            out.append(logits[np.asarray(list(ids), dtype=np.int64)] - lse)
        return out

    def close(self) -> None:
        if getattr(self, "_batch", None) is not None:
            self._lib.llama_batch_free(self._batch)
            self._batch = None
        llm = getattr(self, "llm", None)
        if llm is not None and hasattr(llm, "close"):
            llm.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
