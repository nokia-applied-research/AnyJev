"""The gateway's Decider mode (`--backend strata`), on the offline fake backend: no server, GPU or
transformers. The HTTP contract is Tacit's: POST /v1/decide with one decision or {"items": [...]}."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from anyjev import Decider
from anyjev.backends.fake import FakeBackend
from anyjev.serve import DeciderService, make_server


def _content(state, option):
    # the option whose name the state mentions wins; a deterministic logit otherwise
    return 3.0 if option in state else (sum(map(ord, state + option)) % 7) / 10.0


@pytest.fixture()
def gateway():
    service = DeciderService(Decider(FakeBackend(_content, position_bias=[0.4, 0.1, 0.0, -0.2]), prior="content_free"))
    srv = make_server(service, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", service
    srv.shutdown()


def _post(url, body):
    req = urllib.request.Request(url + "/v1/decide", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


ONE = {"state": "Customer: please refund my order.", "question": "What does the customer want?",
       "options": ["track_order", "refund", "cancel_order"]}


def test_one_decision(gateway):
    url, _ = gateway
    out = _post(url, ONE)
    assert out["answer"] == "refund"
    assert out["level"] == "L0"
    assert set(out["probs"]) == {"track_order", "refund", "cancel_order"}
    assert abs(sum(out["probs"].values()) - 1.0) < 1e-9


def test_a_batch_keeps_its_order(gateway):
    url, _ = gateway
    items = [ONE, {**ONE, "state": "Customer: where is my package? track_order please"}]
    out = _post(url, {"items": items})["decisions"]
    assert [d["answer"] for d in out] == ["refund", "track_order"]


def test_the_same_request_gets_the_same_numbers(gateway):
    # the content-free prior does not move with the traffic: other decisions in between change nothing
    url, _ = gateway
    first = _post(url, ONE)
    for s in ("Customer: cancel_order now", "Customer: track_order", "Customer: refund", "x", "y", "z", "w", "v"):
        _post(url, {**ONE, "state": s})
    assert _post(url, ONE)["probs"] == first["probs"]


def test_yes_no_and_score(gateway):
    url, _ = gateway
    yn = _post(url, {"state": "It is raining.", "question": "Is it raining?", "kind": "yes_no"})
    assert yn["kind"] == "noul" and 0.0 <= yn["probability"] <= 1.0
    sc = _post(url, {"state": "URGENT", "question": "How urgent?", "kind": "score", "options": ["low", "mid", "high"]})
    assert sc["kind"] == "score" and set(sc["probs"]) == {"low", "mid", "high"}


def test_bad_requests_are_400(gateway):
    url, _ = gateway
    for body in ({"question": "q", "options": ["a", "b"]}, {**ONE, "kind": "nope"}, {**ONE, "options": ["a"]},
                 {**ONE, "state": 5}):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(url, body)
        assert e.value.code == 400


def test_stats(gateway):
    url, service = gateway
    _post(url, ONE)
    with urllib.request.urlopen(url + "/v1/stats", timeout=10) as r:
        st = json.load(r)
    assert st["decisions"] == service.count >= 1 and st["prior"] == "content_free" and st["level"] == "L0"
