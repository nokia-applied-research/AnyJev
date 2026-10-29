"""Strata /v1/completions contract tests without a Strata server, a GPU or transformers.

The stub answers with the response shape Strata returns for a ``/v1/completions`` call with
``logprobs=0``, ``allowed_token_ids`` and ``return_tokens_as_token_ids``: the named ids'
full-vocabulary log-probabilities sit in ``choices[0].logprobs.top_logprobs[0]`` keyed
``"token_id:<id>"``, the sampled token among them when it was named.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from anyjev.backends.fake import FakeTokenizer

REAL_SHAPE = {
    "object": "text_completion",
    "choices": [{
        "index": 0, "text": "A", "finish_reason": "length",
        "logprobs": {
            "tokens": ["token_id:41"],
            "token_logprobs": [-0.25],
            # deliberately not in the order the labels were requested: they are read by id
            "top_logprobs": [{"token_id:41": -0.25, "token_id:42": -1.5}],
            "text_offset": [0],
        },
    }],
}


class _Stub(BaseHTTPRequestHandler):
    request_body = None
    response = REAL_SHAPE
    status = 200

    def do_POST(self):  # noqa: N802 - http.server's name
        _Stub.request_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.path = self.path
        payload = json.dumps(_Stub.response).encode()
        self.send_response(_Stub.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture()
def stub():
    _Stub.request_body, _Stub.response, _Stub.status = None, REAL_SHAPE, 200
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _backend(url):
    """A backend with the offline fake tokenizer, so nothing is downloaded and transformers is not
    needed. `FakeTokenizer.encode` maps any text that is not a label to the ids [1, 2]."""
    from anyjev.backends.strata import StrataBackend

    be = StrataBackend.__new__(StrataBackend)
    be.url, be.name, be.api_key = url + "/v1/completions", "served-model", "EMPTY"
    be.workers, be.timeout, be.source = 1, 10.0, "local-tokenizer"
    be.tokenizer = FakeTokenizer()
    return be


def test_scores_the_prompt_ids_with_the_labels_named_and_maps_the_reply_back_by_id(stub):
    got = _backend(stub).next_token_logprobs(["prompt"], [[42, 41]])

    # one token at temperature 0, the labels named, the reply keyed by id: nothing is parsed from text
    assert _Stub.path == "/v1/completions"
    assert _Stub.request_body == {
        "model": "served-model",
        "prompt": [1, 2],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 0,
        "allowed_token_ids": [42, 41],
        "return_tokens_as_token_ids": True,
    }
    np.testing.assert_allclose(got, [[-1.5, -0.25]])
    assert got[0].dtype == np.float64


def test_a_label_the_server_did_not_score_is_an_error(stub):
    with pytest.raises(RuntimeError, match="every requested token id"):
        _backend(stub).next_token_logprobs(["prompt"], [[41, 42, 44]])


def test_a_reply_without_logprobs_is_an_error(stub):
    # what an older Strata (or one on --parallel) gives back: the text, no log-probabilities
    _Stub.response = {"choices": [{"index": 0, "text": "A", "logprobs": None}]}
    with pytest.raises(RuntimeError, match="did not return token log-probabilities"):
        _backend(stub).next_token_logprobs(["prompt"], [[41]])


def test_a_refused_request_says_what_the_server_said(stub):
    _Stub.status, _Stub.response = 400, {"error": {"message": "logprobs is not available ... --parallel"}}
    with pytest.raises(RuntimeError, match="--parallel"):
        _backend(stub).next_token_logprobs(["prompt"], [[41]])


def test_batches_keep_their_order(stub):
    got = _backend(stub).next_token_logprobs(["a", "b"], [[41], [42, 41]])
    np.testing.assert_allclose(got[0], [-0.25])
    np.testing.assert_allclose(got[1], [-1.5, -0.25])


def test_the_url_and_the_tokenizer_source(monkeypatch):
    """`model` is Strata's served name; the tokenizer is loaded from `tokenizer_name`, and the
    base URL may be the server root or its /v1."""
    seen = []

    class _Auto:
        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            seen.append(name)
            return FakeTokenizer()

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = _Auto
    monkeypatch.setitem(sys.modules, "transformers", fake)

    from anyjev.backends.strata import StrataBackend

    for base in ("http://127.0.0.1:1/", "http://127.0.0.1:1/v1"):
        be = StrataBackend(base, "qwen3.8-flash-next-iq3_s", tokenizer_name="Qwen/Qwen3.8-Flash-Next")
        assert (be.name, be.source, be.url) == ("qwen3.8-flash-next-iq3_s", "Qwen/Qwen3.8-Flash-Next",
                                                "http://127.0.0.1:1/v1/completions")
    assert seen == ["Qwen/Qwen3.8-Flash-Next"] * 2


@pytest.mark.engine
def test_real_strata_server_returns_requested_logprobs():
    base_url = os.environ.get("STRATA_BASE_URL")
    if not base_url:
        pytest.skip("set STRATA_BASE_URL to run the Strata engine smoke test")
    from anyjev.backends.strata import StrataBackend

    be = StrataBackend(base_url, os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-iq3_s"),
                       tokenizer_name=os.environ.get("STRATA_TOKENIZER", "Qwen/Qwen3.8-Flash-Next"),
                       api_key=os.environ.get("STRATA_API_KEY", "EMPTY"))
    ids = [be.tokenizer.encode(x, add_special_tokens=False)[0] for x in ("A", "B")]
    prompt = be.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Answer with one letter. Which is a fruit? A) carrot B) apple"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    got = be.next_token_logprobs([prompt], [ids])[0]
    assert got.shape == (2,) and np.all(got <= 0) and got[1] > got[0]
