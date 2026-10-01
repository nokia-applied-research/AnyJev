"""MLX backend tests using a small injected MLX-compatible model."""
import os
import sys
import types

import numpy as np
import pytest


class _Tokenizer:
    pad_token_id = 99
    eos_token_id = 98

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 7 + 1 for c in text]


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return _Array(self.value[key])

    def astype(self, dtype):
        return _Array(self.value.astype(dtype))

    def __sub__(self, other):
        return _Array(self.value - (other.value if isinstance(other, _Array) else other))

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


class _Model:
    def __call__(self, batch):
        x = np.asarray(batch)
        out = np.zeros((*x.shape, 8), dtype=np.float32)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i, j, int(x[i, j]) % 8] = float(j + 1)
        return _Array(out)


@pytest.fixture
def backend(monkeypatch):
    mx = types.ModuleType("mlx.core")
    mx.array = lambda x: _Array(x)
    mx.arange = lambda n: _Array(np.arange(n))
    mx.logsumexp = lambda x, axis, keepdims: _Array(
        np.log(np.exp(np.asarray(x)).sum(axis=axis, keepdims=keepdims)))
    mx.float32 = np.float32
    mx.eval = lambda x: None
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    lm = types.ModuleType("mlx_lm")
    lm.load = lambda *args, **kwargs: (_Model(), _Tokenizer())
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", lm)
    from anyjev.backends.mlx import MLXBackend
    return MLXBackend("fake", batch_size=2)


def test_mlx_batch_padding_and_last_position(backend):
    got = backend.next_token_logprobs(["a", "abcd", "xy"], [[1, 2], [3], [4, 5]])
    assert len(got) == 3
    assert all(x.dtype == np.float64 for x in got)
    assert [x.shape for x in got] == [(2,), (1,), (2,)]
    assert np.all(np.isfinite(np.concatenate(got)))
    # The first row is padded to length four, but its final real position is zero.
    expected = -np.log(np.exp(1.0) + 7.0)
    assert np.allclose(got[0], [expected, expected])


def test_mlx_rejects_empty_prompt(backend):
    with pytest.raises(ValueError, match="empty prompts"):
        backend.next_token_logprobs([""], [[1]])


def test_mlx_checks_lengths(backend):
    with pytest.raises(ValueError, match="same length"):
        backend.next_token_logprobs(["a"], [])


@pytest.mark.engine
def test_mlx_real_engine_smoke():
    try:
        import mlx  # noqa: F401
        import mlx_lm  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"MLX engine is unavailable: {exc}")
    from anyjev.backends.mlx import MLXBackend

    model_name = os.environ.get("ANYJEV_MLX_MODEL")
    if not model_name:
        pytest.skip("set ANYJEV_MLX_MODEL to run the MLX engine smoke test")
    try:
        be = MLXBackend(model_name)
    except Exception as exc:
        pytest.skip(f"MLX model is unavailable: {exc}")
    token = be.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    out = be.next_token_logprobs(["Answer Yes or No."], [[token]])
    assert out[0].shape == (1,)
