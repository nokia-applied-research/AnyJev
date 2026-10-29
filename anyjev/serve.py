"""An HTTP gateway for decisions: Tacit in front of a running `vllm serve`, or the training-free
`Decider` in front of a Strata server.

    vllm serve morriszjm/Tacit-9B --host 127.0.0.1 --port 8000
    python -m anyjev.serve --model morriszjm/Tacit-9B --upstream http://127.0.0.1:8000/v1 --adaptive
    curl -s 127.0.0.1:8100/v1/decide -H 'Content-Type: application/json' \\
        -d '{"state": "Customer: my package has not arrived.", "question": "What does the customer want?",
             "options": ["track_order", "refund"]}'

One `Tacit` serves every client, so the escalation cap (`max_cot_share` over the last `cot_window`
decisions) is shared. POST /v1/decide takes one decision ({"state", "question", "options", "kind"})
or a batch ({"items": [...]}); GET /v1/stats and GET /health. Standard library only; it listens on
127.0.0.1 unless told otherwise and has no authentication of its own.

    python -m anyjev.serve --backend strata --upstream http://127.0.0.1:8080 \\
        --served-model qwen3.8-flash-next-iq3_s --tokenizer Qwen/Qwen3.8-Flash-Next --api-key-env STRATA_API_KEY

`--backend strata` answers the same requests with a `Decider` (L0) over `StrataBackend`. Its default
prior is `content_free`: estimated once per question from probe states, so the same request gets the
same answer however much traffic came before it (a `batch` prior moves with the traffic). With a
Strata that reads canonically (Niko1221/Strata#4604) the answers then repeat exactly. One gateway
reads from one Strata server: two machines round differently, and a decision's reads have to come
from the same one.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
from typing import Any, Dict, List, Optional, Sequence

from anyjev.tacit import Tacit, make_question


class DeciderService:
    """A `Decider` behind the gateway's interface (`decide_batch`, `stats`).  One decision at a time:
    a `Decider` keeps per-question state, and the server behind it answers one request at a time."""

    def __init__(self, decider):
        self.decider = decider
        self.lock = threading.Lock()
        self.count = 0

    def decide_batch(self, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for item in items:
            q = make_question(item["question"], item.get("options"), item.get("kind", "choice"))
            state = item["state"]
            if not isinstance(state, str):
                raise ValueError("state must be a string")
            with self.lock:
                d = next(iter(self.decider.decide(state, [q])))
                self.count += 1
            res = d.to_dict()
            res["probs"] = d.distribution
            out.append(res)
        return out

    @property
    def stats(self) -> Dict[str, Any]:
        return {"decisions": self.count, "level": self.decider.level, "prior": self.decider.prior,
                "backend": getattr(self.decider.backend, "name", "?")}


def make_server(tacit: Tacit, host: str = "127.0.0.1", port: int = 8100) -> http.server.ThreadingHTTPServer:
    """A threading HTTP server answering with `tacit`; call `.serve_forever()` on it."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if self.path == "/v1/stats":
                return self._send(200, tacit.stats)
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/decide":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if "items" in req:
                    return self._send(200, {"decisions": tacit.decide_batch(req["items"])})
                return self._send(200, tacit.decide_batch([req])[0])
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            except RuntimeError as e:                    # the vLLM server failed or is unreachable
                return self._send(502, {"error": str(e)})
            except Exception as e:                        # anything else: an answer, not a dropped connection
                return self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

        def log_message(self, fmt, *args):
            pass

    return http.server.ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m anyjev.serve",
                                 description="Decisions over HTTP: Tacit in front of `vllm serve`, or the Decider "
                                             "in front of Strata.")
    ap.add_argument("--backend", choices=("tacit", "strata"), default="tacit")
    ap.add_argument("--model", default=None, help="tacit: the repo vLLM serves, e.g. morriszjm/Tacit-9B")
    ap.add_argument("--upstream", default=None,
                    help="the server's URL (root or /v1); default http://127.0.0.1:8000/v1 (tacit), "
                         "http://127.0.0.1:8080 (strata)")
    ap.add_argument("--served-model", default=None, help="the served model name (vLLM's --served-model-name; "
                                                          "strata: the name /v1/models lists)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-Flash-Next", help="strata: the model's Hugging Face tokenizer")
    ap.add_argument("--api-key-env", default="STRATA_API_KEY", help="strata: the environment variable holding the key")
    ap.add_argument("--prior", choices=("content_free", "batch", "none"), default="content_free",
                    help="strata: the Decider's label prior (content_free repeats; batch moves with the traffic)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--adaptive", action="store_true", help="send low-confidence decisions to reasoning")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-cot-share", type=float, default=0.2, help="a negative value removes the cap")
    ap.add_argument("--cot-window", type=int, default=1000, help="0 counts every decision since start")
    ap.add_argument("--cot-max-tokens", type=int, default=8192)
    a = ap.parse_args(argv)
    if a.backend == "strata":
        from anyjev import Decider
        from anyjev.backends.strata import StrataBackend

        upstream = a.upstream or "http://127.0.0.1:8080"
        backend = StrataBackend(upstream, a.served_model or "qwen3.8-flash-next-iq3_s", tokenizer_name=a.tokenizer,
                                api_key=os.environ.get(a.api_key_env, "EMPTY"))
        service = DeciderService(Decider(backend, prior=a.prior))
        server = make_server(service, a.host, a.port)
        print("Decider gateway: http://%s:%d/v1/decide -> %s (Strata, prior %s)" % (a.host, a.port, upstream, a.prior),
              flush=True)
        server.serve_forever()
        return
    if not a.model:
        ap.error("--model is required with --backend tacit")
    a.upstream = a.upstream or "http://127.0.0.1:8000/v1"
    tacit = Tacit.from_pretrained(a.model, engine="server", base_url=a.upstream, served_model=a.served_model,
                                  adaptive=a.adaptive, tau=a.tau,
                                  max_cot_share=None if a.max_cot_share < 0 else a.max_cot_share,
                                  cot_window=a.cot_window or None, cot_max_tokens=a.cot_max_tokens)
    server = make_server(tacit, a.host, a.port)
    print("Tacit gateway: http://%s:%d/v1/decide -> %s (%s)" % (a.host, a.port, a.upstream, a.model), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
