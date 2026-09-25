"""LlamaCppBackend against a stub llama_cpp module (no engine, runs in CI), plus one
engine smoke test on a real GGUF when llama-cpp-python and ANYJEV_GGUF are available."""
import ctypes
import hashlib
import os
import sys
import types

import numpy as np
import pytest

from anyjev import Decider, Question
from anyjev.backends.llamacpp import LlamaCppBackend, TokenMismatchError

N_VOCAB = 300   # ids 0..9 reserved (1 = BOS, 2 = EOS), bytes map to 10..265


def _encode(text: str):
    return [b + 10 for b in text.encode("utf-8")]


def _decode(ids):
    special = {1: b"<s>", 2: b"</s>"}
    return b"".join(special.get(i, bytes([i - 10]) if 10 <= i < 266 else b"?") for i in ids)


def reference_logits(tokens):
    """What the stub model outputs after `tokens`: a fixed function of the whole sequence."""
    seed = int(hashlib.md5(repr(list(tokens)).encode()).hexdigest()[:8], 16)
    return np.random.RandomState(seed).randn(N_VOCAB).astype(np.float32) * 3.0


def reference_logprobs(prompt, ids):
    z = reference_logits(_encode(prompt)).astype(np.float64)
    lse = z.max() + np.log(np.exp(z - z.max()).sum())
    return z[list(ids)] - lse


class StubBatch:
    def __init__(self, n):
        self.n_tokens = 0
        self.token = [0] * n
        self.pos = [0] * n
        self.n_seq_id = [0] * n
        self.seq_id = [[0] for _ in range(n)]
        self.logits = [0] * n


class StubLlama:
    """Keeps a KV "memory" and refuses positions that do not continue it, so a missing
    clear between prompts fails loudly. Records which positions asked for logits."""

    def __init__(self, n_ctx=64, n_batch=16, metadata=None):
        self.kwargs = {}
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self.metadata = metadata or {}
        self.memory = []
        self.clears = 0
        self.output_rows = []      # per decode call: number of positions flagged for logits
        self.decode_rc = 0
        self._out = None

    @property
    def ctx(self):
        return self

    def tokenize(self, text, add_bos=True, special=False):
        return ([1] if add_bos else []) + _encode(text.decode("utf-8"))

    def detokenize(self, ids, prev_tokens=None, special=False):
        return _decode(ids)

    def n_vocab(self):
        return N_VOCAB

    def token_bos(self):
        return 1

    def token_eos(self):
        return 2


def make_stub_module(**llama_defaults):
    lib = types.ModuleType("llama_cpp")
    lib.__version__ = "0.3.35"
    lib.created = []

    def Llama(**kw):
        llm = StubLlama(**llama_defaults)   # the stub's own n_ctx=64 / n_batch=16, whatever was asked
        llm.kwargs = kw
        lib.created.append(llm)
        return llm

    def llama_decode(ctx, b):
        if ctx.decode_rc:
            return ctx.decode_rc
        for j in range(b.n_tokens):
            assert b.pos[j] == len(ctx.memory), "positions must continue the cache (was it cleared?)"
            assert b.n_seq_id[j] == 1 and b.seq_id[j][0] == 0
            ctx.memory.append(b.token[j])
        flagged = [j for j in range(b.n_tokens) if b.logits[j]]
        ctx.output_rows.append(len(flagged))
        if flagged:
            vals = reference_logits(ctx.memory)
            ctx._out = (ctypes.c_float * N_VOCAB)(*vals)
        return 0

    def llama_get_logits_ith(ctx, i):
        assert i == -1
        return ctypes.cast(ctx._out, ctypes.POINTER(ctypes.c_float))

    def llama_memory_clear(mem, data):
        mem.memory = []
        mem.clears += 1

    lib.Llama = Llama
    lib.llama_n_ctx = lambda ctx: ctx._n_ctx
    lib.llama_n_batch = lambda ctx: ctx._n_batch
    lib.llama_batch_init = lambda n, embd, n_seq: StubBatch(n)
    lib.llama_batch_free = lambda b: None
    lib.llama_decode = llama_decode
    lib.llama_get_logits_ith = llama_get_logits_ith
    lib.llama_get_memory = lambda ctx: ctx
    lib.llama_memory_clear = llama_memory_clear
    return lib


@pytest.fixture
def stub(monkeypatch):
    lib = make_stub_module()
    monkeypatch.setitem(sys.modules, "llama_cpp", lib)
    return lib


class HFLike:
    """A Hugging Face-shaped tokenizer; `shift` offsets its ids, `label_text` renames label ids."""
    chat_template = None

    def __init__(self, shift=0, label_text=None):
        self.shift = shift
        self.label_text = label_text or {}

    def encode(self, text, add_special_tokens=False):
        return [i + self.shift for i in _encode(text)]

    def decode(self, ids):
        return "".join(self.label_text.get(i, _decode([i - self.shift]).decode()) for i in ids)


def test_full_vocab_logsoftmax_at_the_last_position(stub):
    be = LlamaCppBackend("models/tiny-q8.gguf")
    prompts = ["State: x\nAnswer:", "another prompt"]
    ids = [[_encode("A")[0], _encode("B")[0]], [_encode("Y")[0], _encode("N")[0], 5]]
    out = be.next_token_logprobs(prompts, ids)
    for p, i, lp in zip(prompts, ids, out):
        np.testing.assert_allclose(lp, reference_logprobs(p, i), atol=1e-6)
        assert lp.dtype == np.float64 and np.all(lp < 0)
    assert be.name == "tiny-q8"
    assert stub.created[0].kwargs["logits_all"] is False


def test_only_the_final_token_asks_for_logits_across_chunks(stub):
    be = LlamaCppBackend("m.gguf")                      # stub n_batch = 16
    prompt = "x" * 40                                   # 40 tokens: chunks of 16, 16, 8
    lp = be.next_token_logprobs([prompt], [[20, 21]])[0]
    llm = stub.created[0]
    assert llm.output_rows == [0, 0, 1]
    assert llm.memory == _encode(prompt)
    np.testing.assert_allclose(lp, reference_logprobs(prompt, [20, 21]), atol=1e-6)


def test_cache_is_cleared_before_every_prompt(stub):
    be = LlamaCppBackend("m.gguf")
    a1 = be.next_token_logprobs(["same prompt"], [[30]])[0]
    be.next_token_logprobs(["something else"], [[30]])
    a2 = be.next_token_logprobs(["same prompt"], [[30]])[0]
    assert stub.created[0].clears == 3
    np.testing.assert_array_equal(a1, a2)


def test_prompt_longer_than_context_is_refused(stub):
    be = LlamaCppBackend("m.gguf")                      # stub n_ctx = 64
    with pytest.raises(ValueError, match="raise n_ctx"):
        be.next_token_logprobs(["y" * 65], [[20]])


def test_label_id_outside_vocabulary_is_refused(stub):
    be = LlamaCppBackend("m.gguf")
    with pytest.raises(TokenMismatchError, match="outside"):
        be.next_token_logprobs(["p"], [[N_VOCAB]])


def test_decode_error_is_raised(stub):
    be = LlamaCppBackend("m.gguf")
    stub.created[0].decode_rc = 1
    with pytest.raises(RuntimeError, match="llama_decode returned 1"):
        be.next_token_logprobs(["p"], [[20]])


def test_hf_tokenizer_that_agrees_passes(stub):
    be = LlamaCppBackend("m.gguf", tokenizer=HFLike())
    lp = be.next_token_logprobs(["hello"], [[_encode("A")[0]]])[0]
    np.testing.assert_allclose(lp, reference_logprobs("hello", [_encode("A")[0]]), atol=1e-6)


def test_hf_tokenizer_that_disagrees_on_a_prompt_is_refused(stub):
    be = LlamaCppBackend("m.gguf", tokenizer=HFLike(shift=1))
    with pytest.raises(TokenMismatchError, match="first difference at token 0"):
        be.next_token_logprobs(["hello"], [[20]])


def test_label_id_naming_different_text_is_refused(stub):
    a = _encode("A")[0]
    ok = LlamaCppBackend("m.gguf", tokenizer=HFLike(label_text={a: " A"}))   # whitespace only: accepted
    ok.next_token_logprobs(["hello"], [[a]])
    bad = LlamaCppBackend("m.gguf", tokenizer=HFLike(label_text={a: "Z"}))
    with pytest.raises(TokenMismatchError, match="label token id"):
        bad.next_token_logprobs(["hello"], [[a]])


def test_gguf_chat_template_is_rendered_like_transformers(monkeypatch):
    pytest.importorskip("jinja2")   # a llama-cpp-python dependency, not an anyjev one
    tpl = ("{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}\n{% endfor %}"
           "{% if add_generation_prompt %}<assistant>{% if enable_thinking is defined and not enable_thinking %}"
           "<nothink>{% endif %}{% endif %}")
    lib = make_stub_module(metadata={"tokenizer.chat_template": tpl})
    monkeypatch.setitem(sys.modules, "llama_cpp", lib)
    tok = LlamaCppBackend("m.gguf").tokenizer
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert out == "<s><system>S\n<user>U\n<assistant><nothink>"
    assert tok.encode("AB") == _encode("AB")


def test_decider_end_to_end_on_the_stub(monkeypatch):
    stub = make_stub_module(n_ctx=1024, n_batch=128)
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)
    d = Decider(LlamaCppBackend("m.gguf"))
    q = Question.choice("Which team?", ["billing", "technical", "sales"], name="route")
    dec = d.decide("My card was charged twice.", [q], level="L0")["route"]
    assert dec.level == "L0"
    assert abs(float(dec.probs.sum()) - 1.0) < 1e-9
    assert stub.created[0].clears == 3                  # one forward per cyclic shift


def test_old_llama_cpp_python_is_refused(monkeypatch):
    lib = make_stub_module()
    lib.__version__ = "0.3.2"
    monkeypatch.setitem(sys.modules, "llama_cpp", lib)
    with pytest.raises(ImportError, match="llama-cpp-python>=0.3.16"):
        LlamaCppBackend("m.gguf")


@pytest.mark.engine
def test_engine_smoke_on_a_real_gguf():
    pytest.importorskip("llama_cpp")
    path = os.environ.get("ANYJEV_GGUF")
    if not path:
        pytest.skip("set ANYJEV_GGUF to a local .gguf file to run the engine smoke test")
    be = LlamaCppBackend(path, n_ctx=2048)
    q = Question.choice("Which team handles this?", ["billing", "technical", "sales", "other"], name="route")
    dec = Decider(be).decide("My card was charged twice.", [q], level="L0")["route"]
    assert dec.level == "L0" and abs(float(dec.probs.sum()) - 1.0) < 1e-6
