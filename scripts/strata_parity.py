"""Backend parity: StrataBackend vs the same server's unrestricted top-20, then one L0 decision set.

    python scripts/strata_parity.py --base-url http://127.0.0.1:8080 \
        --model qwen3.8-flash-next-iq3_s --tokenizer Qwen/Qwen3.8-Flash-Next --api-key ...

Strata serves models (Qwen3.8-Flash-Next, 125B) that HFBackend cannot load on the same PC, so the
reference is the server's other path: the prompt sent as text (Strata tokenizes it) with
``logprobs=20`` and no ``allowed_token_ids``. That checks the Hugging Face tokenizer against the
engine's vocabulary and the named-id readout against the top-K one; labels outside the top 20 are
not compared. Then a Decider at L0 runs over the same questions and prints its decisions.
Checked on Strata 0.1.53 with canonical reads (Niko1221/Strata#4604), Qwen3.8-Flash-Next IQ3_S,
launch defaults, on an RTX 5080 and a GTX 1080 Ti: 50 of 50 labels in the top 20, and the two paths
agree exactly (max_abs_diff 0.000000) - both reads are one token's logprobs, so both are canonical.
On a Strata without it, reads drift between requests (Niko1221/Strata#4059) and the difference
measured 0.62 to 1.14: run-to-run spread, not a readout error.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

from anyjev import Decider, Question
from anyjev.readout import answer_labels, build_prompt, label_ids_for_perm, map_label_tokens, render_chat
from anyjev.state import render_state


def top20(backend, prompt):
    body = {"model": backend.name, "prompt": prompt, "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
            "return_tokens_as_token_ids": True}
    req = urllib.request.Request(backend.url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {backend.api_key}"})
    with urllib.request.urlopen(req, timeout=backend.timeout) as r:
        top = json.load(r)["choices"][0]["logprobs"]["top_logprobs"][0]
    return {int(k.split(":", 1)[1]): v for k, v in top.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen3.8-flash-next-iq3_s")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()
    from anyjev.backends.strata import StrataBackend

    be = StrataBackend(args.base_url, args.model, tokenizer_name=args.tokenizer, api_key=args.api_key)
    qs = [Question.choice("Which team handles this?", ["billing", "technical", "sales", "other"], name="team"),
          Question.noul("Is this message a complaint?", name="complaint"),
          Question.score("How urgent is this?", levels=["not", "low", "medium", "high"], name="urgency")]
    states = ["My card was charged twice.", "The app crashes on login.", "Do you offer bulk discounts?",
              "Thanks, all sorted now!", "URGENT: production is down for all users."]
    prompts, ids = [], []
    for q in qs:
        base = map_label_tokens(be.tokenizer, answer_labels(q))
        perm = list(range(q.k))
        for state in states:
            prompts.append(render_chat(be.tokenizer, build_prompt(render_state(state), q, perm)))
            ids.append(label_ids_for_perm(q, base, perm))
    t0 = time.time()
    named = be.next_token_logprobs(prompts, ids)
    t_named = time.time() - t0
    diffs, compared, outside = [], 0, 0
    for prompt, label_ids, row in zip(prompts, ids, named):
        ref = top20(be, prompt)
        for i, v in zip(label_ids, row):
            if i in ref:
                diffs.append(abs(ref[i] - v))
                compared += 1
            else:
                outside += 1
    print(f"prompts={len(prompts)} labels_compared={compared} outside_top20={outside} "
          f"max_abs_diff={max(diffs):.6f} named_read={t_named:.1f}s")

    d = Decider(be)
    t0 = time.time()
    for state in states:
        out = d.decide(state, qs)
        print(f"{state[:40]!r:44}", "  ".join(
            f"{q.name}={out[q.name].answer}({max(out[q.name].distribution.values()):.2f},{out[q.name].level})"
            for q in qs))
    print(f"decider: {len(states)} states x {len(qs)} questions at L0 in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
