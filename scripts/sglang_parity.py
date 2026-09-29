"""Backend parity: SGLangBackend vs HFBackend on the same prompts.

    python scripts/sglang_parity.py --model Qwen/Qwen2.5-7B-Instruct \
        --base-url http://127.0.0.1:30000

Start SGLang first:
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000

The script compares the requested token log-probabilities without generating a
new token. SGLang's native ``/generate`` endpoint scores the prompt's appended
placeholder position with ``token_ids_logprob``.
"""
from __future__ import annotations

import argparse

import numpy as np

from anyjev import Question
from anyjev.readout import answer_labels, build_prompt, label_ids_for_perm, map_label_tokens, render_chat
from anyjev.state import render_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--device", default=None, help="HF device; defaults to CUDA when available, else CPU")
    ap.add_argument("--dtype", default=None, help="HF dtype; defaults to bfloat16 on CUDA, else float32")
    args = ap.parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bfloat16" if device == "cuda" else "float32")
    from anyjev.backends.hf import HFBackend
    from anyjev.backends.sglang import SGLangBackend

    hf = HFBackend(args.model, device=device, dtype=dtype, batch_size=8)
    sgl = SGLangBackend(args.base_url, args.model)
    qs = [Question.choice("Which team handles this?", ["billing", "technical", "sales", "other"]),
          Question.noul("Is this message a complaint?"),
          Question.score("How urgent is this?", levels=["not", "low", "medium", "high"])]
    states = ["My card was charged twice.", "The app crashes on login.", "Do you offer bulk discounts?",
              "Thanks, all sorted now!", "URGENT: production is down for all users."]
    prompts, ids = [], []
    for q in qs:
        base = map_label_tokens(hf.tokenizer, answer_labels(q))
        perm = list(range(q.k))
        for state in states:
            prompts.append(render_chat(hf.tokenizer, build_prompt(render_state(state), q, perm)))
            ids.append(label_ids_for_perm(q, base, perm))
    expected = hf.next_token_logprobs(prompts, ids)
    actual = sgl.next_token_logprobs(prompts, ids)
    diffs, agree = [], 0
    for left, right in zip(expected, actual):
        diffs.append(float(np.max(np.abs(left - right))))
        agree += int(np.argmax(left) == np.argmax(right))
    print(f"prompts={len(prompts)} max_abs_diff={max(diffs):.4f} "
          f"mean={np.mean(diffs):.4f} argmax_agreement={agree}/{len(prompts)}")
    print("sglang sample:", np.round(actual[0], 3), "hf:", np.round(expected[0], 3))


if __name__ == "__main__":
    main()
