"""SGLang native API contract tests without requiring a SGLang install or GPU."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest


class _Stub(BaseHTTPRequestHandler):
    request_body = None

    def do_POST(self):  # noqa: N802 - http.server's name
        _Stub.request_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = {"input_token_ids_logprobs_val": [[-0.25, -1.5]]}
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture()
def stub():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_native_generate_scores_requested_prompt_token_ids(stub, monkeypatch):
    pytest.importorskip("transformers")
    import transformers

    from anyjev.backends.sglang import SGLangBackend

    class _Tok:
        pad_token_id = 99
        eos_token_id = 98

        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            return _Tok()

        def encode(self, text, add_special_tokens=False):
            return [10, 11, 12]

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tok)
    backend = SGLangBackend(stub, "served-model", tokenizer_name="local-tokenizer", workers=1)
    got = backend.next_token_logprobs(["prompt"], [[41, 42]])

    np.testing.assert_allclose(got, [[-0.25, -1.5]])
    assert _Stub.request_body == {
        "input_ids": [10, 11, 12, 99],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": 3,
        "token_ids_logprob": [41, 42],
    }


def test_missing_requested_logprob_is_an_error(stub, monkeypatch):
    pytest.importorskip("transformers")
    import transformers

    from anyjev.backends.sglang import SGLangBackend

    class _Tok:
        pad_token_id = 0
        eos_token_id = None

        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            return _Tok()

        def encode(self, text, add_special_tokens=False):
            return [10]

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tok)
    backend = SGLangBackend(stub, "served-model", tokenizer_name="local-tokenizer", workers=1)
    with pytest.raises(RuntimeError, match="every requested token id"):
        backend.next_token_logprobs(["prompt"], [[41, 42, 43]])


@pytest.mark.engine
def test_real_sglang_server_returns_requested_logprobs():
    base_url = os.environ.get("SGLANG_BASE_URL")
    if not base_url:
        pytest.skip("set SGLANG_BASE_URL to run the SGLang engine smoke test")
    model = os.environ.get("SGLANG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    from anyjev.backends.sglang import SGLangBackend

    backend = SGLangBackend(base_url, model)
    ids = [backend.tokenizer.encode(" A", add_special_tokens=False)[-1],
           backend.tokenizer.encode(" B", add_special_tokens=False)[-1]]
    got = backend.next_token_logprobs(["Answer with one letter:"], [ids])[0]
    assert got.shape == (2,) and np.all(np.isfinite(got))
