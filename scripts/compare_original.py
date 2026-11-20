#!/usr/bin/env python3
"""Compare littlebit_ref against the original implementations.

  --mode dense      our Qwen3Ref with dense HF weights  vs  transformers (checks the skeleton)
  --mode littlebit  our Qwen3Ref with LittleBit factors vs LittleBit's load_quantized_model
                    (imports LittleBit from --littlebit-dir, CC BY-NC 4.0, nothing copied)

Pass/fail is decided on logits only (littlebit_ref.metrics), not on argmax/text.
Both sides run on CPU. Default dtype float32; --dtype float64 shows the pure implementation
difference without rounding noise. Prints per-layer and logits error statistics as JSON.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from littlebit_ref import Qwen3Ref, load_checkpoint  # noqa: E402
from littlebit_ref.prompts import PROMPTS  # noqa: E402
from littlebit_ref.metrics import TOL_FP_REFERENCE, logits_stats, tensor_stats as stats, within  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "littlebit"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--littlebit-dir")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--out", help="write JSON results here")
    ap.add_argument("--emulate-bf16-l", action="store_true",
                    help="round l=v1*u2 to BF16 like LittleBit's loader does")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.mode == "dense":
        from transformers import AutoModelForCausalLM
        orig = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
        sd = {k: v.to(dtype) for k, v in orig.state_dict().items()}
        cfg = json.loads((Path(args.model) / "config.json").read_text())
        lin = {k[: -len(".weight")]: v for k, v in sd.items()
               if any(k.endswith(p + ".weight") for p in Qwen3Ref.PROJ)}
        ref = Qwen3Ref(cfg, sd, lin)
    else:
        sys.path.insert(0, str(Path(args.littlebit_dir).resolve()))
        from quantization.utils.quant_util import load_quantized_model  # LittleBit
        lbc = json.loads((Path(args.model) / "littlebit_config.json").read_text())
        qargs = argparse.Namespace(**lbc, model_id=args.model)
        orig = load_quantized_model(args.model, qargs, torch_dtype=dtype, device="cpu")
        orig.config._attn_implementation = "eager"
        ref = Qwen3Ref.from_checkpoint(load_checkpoint(args.model, dtype=dtype, l_round=torch.bfloat16 if args.emulate_bf16_l else None))
    orig.eval()

    results = {"mode": args.mode, "dtype": args.dtype, "emulate_bf16_l": args.emulate_bf16_l, "prompts": []}
    n = ref.n_layer
    for text in PROMPTS:
        ids = tok(text, return_tensors="pt").input_ids[0]
        hs = {}
        logits_ref = ref.forward(ids, hook=lambda k, t: hs.__setitem__(k, t))
        with torch.no_grad():
            o = orig(ids[None], output_hidden_states=True, use_cache=False)
        layers = [stats(o.hidden_states[i + 1][0], hs[f"model.layers.{i}.out"]) for i in range(n - 1)]
        layers.append(stats(o.hidden_states[n][0], hs["norm"]))  # HF applies final norm to the last one
        r = {"text": text, "n_tokens": len(ids),
             "logits": logits_stats(o.logits[0], logits_ref),
             "worst_layer_rel_rms": max(l["rel_rms"] for l in layers),
             "pass": None,
             "layers": layers}
        r["pass"] = within(r["logits"], TOL_FP_REFERENCE)
        results["prompts"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "layers"}))
    results["pass"] = all(p["pass"] for p in results["prompts"])
    print("PASS" if results["pass"] else "FAIL", "(criteria:", TOL_FP_REFERENCE, "on logits)")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
