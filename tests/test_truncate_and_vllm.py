"""Truncation and the vLLM backend, without a GPU or a server.

Truncation is exercised on a tiny randomly-initialised model, which is enough: what can go
wrong is which tensors are kept and what the config then says, not arithmetic. The vLLM backend
is exercised against a stub HTTP server, because the thing worth pinning is its contract --
which layer indices it accepts, what it refuses, and where it reads the config from -- and all
three have already been got wrong once.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from anyjev.truncate import layer_of


def test_layer_of_finds_the_block_a_tensor_belongs_to():
    assert layer_of("model.layers.7.self_attn.q_proj.weight") == 7
    assert layer_of("layers.13.input_layernorm.weight") == 13
    assert layer_of("model.embed_tokens.weight") is None
    assert layer_of("model.norm.weight") is None
    assert layer_of("lm_head.weight") is None
    # a number elsewhere in the name is not a block index
    assert layer_of("model.embed_tokens_2.weight") is None


def test_truncate_keeps_the_first_blocks_and_says_so(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model("llama", hidden_size=32, intermediate_size=64,
                               num_hidden_layers=6, num_attention_heads=4,
                               num_key_value_heads=4, vocab_size=64, max_position_embeddings=64)
    src = tmp_path / "src"
    AutoModelForCausalLM.from_config(cfg).save_pretrained(src)
    AutoConfig.from_pretrained(src).save_pretrained(src)

    from anyjev.truncate import truncate
    out = tmp_path / "out"
    truncate(str(src), 4, str(out))

    kept = AutoConfig.from_pretrained(out)
    assert kept.num_hidden_layers == 4
    meta = json.loads((out / "anyjev_truncation.json").read_text())
    assert meta["kept_blocks"] == 4 and meta["original_blocks"] == 6
    assert meta["dropped_tensors"] > 0

    from safetensors.torch import load_file
    names = list(load_file(str(out / "model.safetensors")))
    assert not [n for n in names if (layer_of(n) or 0) >= 4]
    assert any(layer_of(n) == 3 for n in names)          # the last kept block is there
    assert any(layer_of(n) is None for n in names)       # embeddings / norm / head kept
    # and it loads as an ordinary model (no dtype keyword here on purpose: its name differs
    # between transformers 4 and 5, which is the very thing HFBackend has to paper over)
    m = AutoModelForCausalLM.from_pretrained(out)
    assert m.config.num_hidden_layers == 4
    assert len(m.model.layers) == 4

    with pytest.raises(FileExistsError):
        truncate(str(src), 3, str(out))
    with pytest.raises(ValueError):
        truncate(str(src), 99, str(tmp_path / "nope"))


class _Stub(BaseHTTPRequestHandler):
    """Answers /v1/embeddings with a vector derived from the prompt length."""

    dim = 8

    def do_POST(self):                                    # noqa: N802 - http.server's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        vec = [float(len(body["input"]) % 7) + i for i in range(self.dim)]
        payload = json.dumps({"data": [{"embedding": vec}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):                            # keep the test output clean
        pass


@pytest.fixture()
def stub():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _backend(url, monkeypatch):
    """A backend whose tokenizer load is stubbed. `VLLMBackend.__init__` imports
    `AutoTokenizer` from `transformers` at call time, so the patch has to land there rather
    than on the backend module."""
    import transformers

    from anyjev.backends.vllm import VLLMBackend

    class _Tok:
        @staticmethod
        def from_pretrained(name, *a, **kw):
            return object()

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tok)
    be = VLLMBackend(url, "served-alias", tokenizer_name="some/real-model")
    be._n_layers = 28                                     # as if read from the config
    return be


def test_hidden_states_returns_one_layer_and_refuses_the_others(stub, monkeypatch):
    pytest.importorskip("transformers")
    be = _backend(stub, monkeypatch)
    feats, lps, pos = be.hidden_states(["a", "bb"], [-1])
    assert feats.shape == (2, 1, _Stub.dim) and pos is None and lps == [None, None]
    assert np.allclose(be.hidden_states(["a"], [28])[0], be.hidden_states(["a"], [-1])[0])

    # a head fit at block 18 must not be served the final block in silence
    with pytest.raises(ValueError, match="only the final layer"):
        be.hidden_states(["a"], [18])
    with pytest.raises(ValueError, match="no logits"):
        be.hidden_states(["a"], [-1], token_ids=[[1, 2]])
    with pytest.raises(ValueError, match="one vector per prompt"):
        be.hidden_states(["a"], [-1], positions=[[0]])


def test_the_config_comes_from_the_source_not_the_served_alias(stub, monkeypatch):
    """`--served-model-name` is an API label; tokenizer and config have to come from a path."""
    pytest.importorskip("transformers")
    be = _backend(stub, monkeypatch)
    assert be.name == "served-alias"
    assert be.source == "some/real-model"


class _Completions(BaseHTTPRequestHandler):
    """Answers /v1/completions with a fixed `top_logprobs` entry, keyed by token string."""

    top: dict = {}

    def do_POST(self):                                    # noqa: N802 - http.server's name
        self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.dumps({"choices": [{"logprobs": {"top_logprobs": [self.top]}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


class _IdTok:
    """Renders token id i as "<i>", both ways the backend looks a label up."""

    @staticmethod
    def decode(ids):
        return f"<{ids[0]}>"

    @staticmethod
    def convert_ids_to_tokens(i):
        return f"<{i}>"


@pytest.fixture()
def completions(monkeypatch):
    def serve(top):
        handler = type("H", (_Completions,), {"top": top})
        srv = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        be = _backend(f"http://127.0.0.1:{srv.server_port}", monkeypatch)
        be.tokenizer = _IdTok()
        return be

    servers = []
    yield serve
    for srv in servers:
        srv.shutdown()


def test_next_token_logprobs_reads_each_label_by_id(completions):
    pytest.importorskip("transformers")
    be = completions({"<2>": -2.5, "<1>": -0.1})
    (row,) = be.next_token_logprobs(["p"], [[1, 2]])
    assert np.allclose(row, [-0.1, -2.5])


def test_a_label_missing_from_top_logprobs_is_an_error(completions):
    """vLLM's default `--logprobs-mode raw_logprobs` computes logprobs before `allowed_token_ids`
    masks the vocabulary, so the top-K can hold a non-label token in place of a label. A label
    filled in with a constant would be a confident wrong answer with no symptom."""
    pytest.importorskip("transformers")
    from anyjev.readout import LabelTokenError

    be = completions({"<1>": -0.1, "<9>": -1.0})          # label 2 fell out, token 9 took its place
    with pytest.raises(LabelTokenError, match=r"\[2\].*processed_logprobs"):
        be.next_token_logprobs(["p"], [[1, 2]])
