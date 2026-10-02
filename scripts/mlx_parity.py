"""Compare direct MLX-LM and Transformers readouts on identical prompts.

Example: ``python scripts/mlx_parity.py --hf-model Qwen/Qwen2.5-3B-Instruct
--mlx-model mlx-community/Qwen2.5-3B-Instruct-4bit``.
"""
import argparse

import numpy as np

from anyjev import Question
from anyjev.backends.hf import HFBackend
from anyjev.readout import answer_labels, build_prompt, label_ids_for_perm, map_label_tokens, render_chat
from anyjev.state import render_state


def _restricted_lsm(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    values = values - values.max()
    return values - np.log(np.exp(values).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--mlx-model", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    hf = HFBackend(args.hf_model, batch_size=args.batch_size)
    from anyjev.backends.mlx import MLXBackend
    mlx = MLXBackend(args.mlx_model, batch_size=args.batch_size)
    qs = [Question.choice("Which team handles this?", ["billing", "technical", "sales", "other"]),
          Question.noul("Is this message a complaint?"),
          Question.score("How urgent is this?", levels=["not", "low", "medium", "high"])]
    states = ["My card was charged twice.", "The app crashes on login.", "Do you offer bulk discounts?",
              "Thanks, all sorted now!", "URGENT: production is down for all users."]
    prompts, ids = [], []
    for q in qs:
        labels = answer_labels(q)
        hf_base = map_label_tokens(hf.tokenizer, labels)
        mlx_base = map_label_tokens(mlx.tokenizer, labels)
        if hf_base != mlx_base:
            raise SystemExit(f"label tokenization mismatch for {labels}: HF={hf_base}, MLX={mlx_base}")
        perm = list(range(q.k))
        for state in states:
            text = build_prompt(render_state(state), q, perm)
            hp = render_chat(hf.tokenizer, text)
            mp = render_chat(mlx.tokenizer, text)
            h_ids = hf.tokenizer.encode(hp, add_special_tokens=False)
            m_ids = mlx.tokenizer.encode(mp, add_special_tokens=False)
            if h_ids != m_ids:
                raise SystemExit("prompt tokenization mismatch; parity comparison is invalid")
            prompts.append((hp, mp))
            ids.append(label_ids_for_perm(q, hf_base, perm))
    hf_values = hf.next_token_logprobs([x[0] for x in prompts], ids)
    mlx_values = mlx.next_token_logprobs([x[1] for x in prompts], ids)
    diffs = [float(np.max(np.abs(_restricted_lsm(a) - _restricted_lsm(b)))) for a, b in zip(hf_values, mlx_values)]
    agree = sum(int(np.argmax(a) == np.argmax(b)) for a, b in zip(hf_values, mlx_values))
    print(f"prompts={len(prompts)} max_abs_diff_restricted_logsoftmax={max(diffs):.4f} "
          f"mean={np.mean(diffs):.4f} argmax_agreement={agree}/{len(prompts)}")


if __name__ == "__main__":
    main()
