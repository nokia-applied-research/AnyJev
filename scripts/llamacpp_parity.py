"""Backend parity: LlamaCppBackend (GGUF) vs HFBackend on the same prompts.
    python convert_hf_to_gguf.py <HF snapshot dir> --outtype f32 --outfile qwen2.5-0.5b-instruct-f32.gguf
    python scripts/llamacpp_parity.py --gguf qwen2.5-0.5b-instruct-f32.gguf --model Qwen/Qwen2.5-0.5B-Instruct \
        --device cpu --dtype float32 --json parity.json
Both sides get the prompts rendered by the Hugging Face tokenizer and the same label ids;
the llama.cpp side checks every prompt tokenizes identically in the GGUF (TokenMismatchError
otherwise). Reports max |diff| of the full-vocabulary log-probs of the label tokens, of the
log-softmax restricted to the labels, and argmax agreement over choice / noul / score prompts.
For parity use an F32 GGUF converted from the same Hugging Face snapshot the HF side loads: an F16
or quantized file also measures llama.cpp's reduced-precision compute (on Qwen2.5-0.5B the official
F16 GGUF differs from fp32 transformers by up to ~1.2 nats, the F32 conversion by ~0.02).
"""
import argparse
import json
import platform

import numpy as np

from anyjev import Question
from anyjev.readout import build_prompt, label_ids_for_perm, render_chat, resolve_labels
from anyjev.state import render_state


def lsm(x):
    x = np.asarray(x, float)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="path to the GGUF file")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="the HF model the GGUF was converted from")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the numbers and the environment here")
    args = ap.parse_args()
    import llama_cpp
    import torch
    import transformers

    from anyjev.backends.hf import HFBackend
    from anyjev.backends.llamacpp import LlamaCppBackend
    hf = HFBackend(args.model, device=args.device, dtype=args.dtype, batch_size=8)
    lc = LlamaCppBackend(args.gguf, tokenizer=hf.tokenizer, n_gpu_layers=args.n_gpu_layers)
    qs = [Question.choice("Which team handles this?", ["billing", "technical", "sales", "other"]),
          Question.noul("Is this message a complaint?"),
          Question.score("How urgent is this?", levels=["not", "low", "medium", "high"])]
    states = ["My card was charged twice.", "The app crashes on login.", "Do you offer bulk discounts?",
              "Thanks, all sorted now!", "URGENT: production is down for all users."]
    prompts, ids = [], []
    for q in qs:
        labels, base = resolve_labels(hf.tokenizer, q)
        perm = list(range(q.k))
        for s in states:
            prompts.append(render_chat(hf.tokenizer, build_prompt(render_state(s), q, perm, labels=labels)))
            ids.append(label_ids_for_perm(q, base, perm))
    a = hf.next_token_logprobs(prompts, ids)
    b = lc.next_token_logprobs(prompts, ids)
    full = [float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a, b)]
    restricted = [float(np.max(np.abs(lsm(x) - lsm(y)))) for x, y in zip(a, b)]
    agree = sum(int(np.argmax(x) == np.argmax(y)) for x, y in zip(a, b))
    print(f"prompts={len(prompts)} argmax_agreement={agree}/{len(prompts)}")
    print(f"max_abs_diff_full_vocab_logprob={max(full):.4f} mean={np.mean(full):.4f}")
    print(f"max_abs_diff_restricted_logsoftmax={max(restricted):.4f} mean={np.mean(restricted):.4f}")
    print("llama.cpp sample:", np.round(b[0], 3), "hf:", np.round(a[0], 3))
    if args.json:
        out = {"gguf": args.gguf, "model": args.model, "hf_device": args.device, "hf_dtype": args.dtype,
               "n_gpu_layers": args.n_gpu_layers, "prompts": len(prompts), "argmax_agreement": agree,
               "max_abs_diff_full_vocab_logprob": max(full), "mean_abs_diff_full_vocab_logprob": float(np.mean(full)),
               "max_abs_diff_restricted": max(restricted), "mean_abs_diff_restricted": float(np.mean(restricted)),
               "env": {"platform": platform.platform(), "processor": platform.processor(),
                       "python": platform.python_version(), "llama_cpp_python": llama_cpp.__version__,
                       "torch": torch.__version__, "transformers": transformers.__version__}}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
