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

On Windows, the default PyPI wheel is CPU-only and works; a prebuilt CUDA wheel from
llama-cpp-python's own extra index has been seen to crash at context creation (`illegal
instruction`, `WinError -1073741795`) on CPUs without AVX-512, which is most consumer
CPUs. If that happens, use the default wheel or build with `-DGGML_NATIVE=OFF`.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# llama_get_memory / llama_memory_clear / llama_batch_init / llama_get_logits_ith(-1) were
# added around 0.3.16. Checked by symbol, not by parsing __version__: the prebuilt CUDA and
# Metal wheels carry version strings like "0.3.16+cu124" that a naive int-parse rejects even
# though they have every symbol, while a real pre-0.3.16 build is missing them regardless of
# what its version string says.
_REQUIRED_SYMBOLS = ("llama_get_memory", "llama_memory_clear", "llama_batch_init", "llama_batch_free",
                     "llama_get_logits_ith", "llama_decode", "llama_n_ctx", "llama_n_batch")


class TokenMismatchError(ValueError):
    """The Hugging Face tokenizer and the GGUF vocabulary disagree on a prompt or a label id."""


def _import_llama_cpp():
    try:
        import llama_cpp
    except ImportError as e:
        raise ImportError('LlamaCppBackend needs llama-cpp-python: pip install "anyjev[llamacpp]"') from e
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(llama_cpp, name)]
    if missing:
        raise ImportError(
            "LlamaCppBackend needs a llama-cpp-python build exposing "
            f"{', '.join(missing)} (present from ~0.3.16 on, including the default PyPI "
            f"wheel); found {getattr(llama_cpp, '__version__', '?')} without them")
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
        self._formatters: Dict[bool, Any] = {}   # add_generation_prompt -> cached Jinja2ChatFormatter

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
        """Render the GGUF chat template through `llama_cpp.llama_chat_format.Jinja2ChatFormatter`
        — the same sandboxed Jinja environment `Llama()` itself builds from this GGUF's metadata
        (trim_blocks / lstrip_blocks, the `tojson` filter, `raise_exception`, `tools` / `documents`
        bound to `None` rather than left undefined), plus its pass-through for the `{% generation %}`
        tag (used by e.g. SmolLM3's template) that a bare sandboxed environment cannot parse. The
        formatter is built once per `add_generation_prompt` value and reused, not recompiled per call.
        """
        if not self.chat_template:
            raise ValueError("this GGUF has no tokenizer.chat_template")
        fmt = self._formatters.get(add_generation_prompt)
        if fmt is None:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            fmt = Jinja2ChatFormatter(template=self.chat_template, bos_token=self.bos_token,
                                      eos_token=self.eos_token, add_generation_prompt=add_generation_prompt)
            self._formatters[add_generation_prompt] = fmt
        text = fmt(messages=messages, **kwargs).prompt
        return self.encode(text) if tokenize else text


class LlamaCppBackend:
    """`name` is the artifact-identity key: `Decider.load_artifact` / `load_artifacts` check
    `backend.name`, not the model file's path, so it defaults from the GGUF's own
    `general.basename` / `general.name` metadata rather than the filename — `llama-quantize`
    writes `ggml-model-Q4_K_M.gguf` for every model alike, so two different quantized models
    in different directories would otherwise collide under the same default name and silently
    accept each other's L1 calibration. Pass `name=` explicitly to be certain.

    Calling any method after `close()` raises `RuntimeError` rather than segfaulting.

    `Decider.observe()` for L2 will raise once the first labelled example reaches `fit_head`,
    because this backend exposes no hidden states; `level="auto"` and `level="L2"` both
    degrade to L1 cleanly instead, so only the manual, L2-only label-collection workflow is
    affected.
    """

    def __init__(self, model_path: str, tokenizer: Any = None, name: Optional[str] = None,
                 n_ctx: int = 4096, n_batch: int = 512, n_gpu_layers: int = 0,
                 verbose: bool = False, **llama_kwargs: Any):
        lib = _import_llama_cpp()
        self._lib = lib
        self.llm = lib.Llama(model_path=model_path, n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=n_gpu_layers,
                             logits_all=False, verbose=verbose, **llama_kwargs)
        meta = getattr(self.llm, "metadata", None) or {}
        default_name = (meta.get("general.basename") or meta.get("general.name")
                        or os.path.splitext(os.path.basename(model_path))[0])
        self.name = name or default_name
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
        # data=False: only the cells need resetting. Every prompt decodes into fresh cells
        # from position 0, so the KV data buffer itself is never read; data=True would memset
        # it anyway (~0.5 GB per prompt at n_ctx=4096 on a 7-8B GGUF for nothing read back).
        self._lib.llama_memory_clear(self._lib.llama_get_memory(self.llm.ctx), False)

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
        if self.llm is None:
            raise RuntimeError(f"{self.name}: this LlamaCppBackend is closed")
        # Materialize both up front: a generator of ids would otherwise be drained by
        # _check_label_ids below and come back empty when read a second time, and zip()
        # would silently truncate to the shorter of the two rather than raising.
        prompts = list(prompts)
        token_ids = [list(ids) for ids in token_ids]
        if len(prompts) != len(token_ids):
            raise ValueError(
                f"got {len(prompts)} prompts but {len(token_ids)} label-id lists; they must pair up")
        out: List[np.ndarray] = []
        for prompt, ids in zip(prompts, token_ids):
            self._check_label_ids(ids)
            logits = self._last_row_logits(self._prompt_tokens(prompt))
            m = logits.max()
            lse = m + np.log(np.exp(logits - m).sum())       # full-vocabulary log-softmax
            out.append(logits[np.asarray(ids, dtype=np.int64)] - lse)
        return out

    def close(self) -> None:
        if getattr(self, "_batch", None) is not None:
            self._lib.llama_batch_free(self._batch)
            self._batch = None
        llm = getattr(self, "llm", None)
        if llm is not None:
            if hasattr(llm, "close"):
                llm.close()
            self.llm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
