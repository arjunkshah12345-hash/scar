"""Analytic persistent streaming-state accounting for Study 2.

This counts runtime tensors needed to continue processing one stream after a
prefix. Parameters are deliberately excluded. It is a protocol accounting
tool, not a claim about allocator overhead or framework implementation details.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FLOAT_BYTES = 4


def persistent_state_bytes(model: str, context: int) -> int:
    if context <= 0:
        raise ValueError("context must be positive")
    if model == "elman":
        return 192 * FLOAT_BYTES
    if model == "gru":
        return 112 * FLOAT_BYTES
    if model == "lstm":
        return 2 * 96 * FLOAT_BYTES
    if model == "scar":
        return (88 + 16 * 88) * FLOAT_BYTES
    if model == "scar_carrier":
        return 112 * FLOAT_BYTES
    if model == "scar_norecall":
        return (98 + 16 * 98) * FLOAT_BYTES
    if model == "transformer":
        # Three full-attention layers, each retaining K and V for every token.
        return 3 * 2 * context * 56 * FLOAT_BYTES
    if model == "token_merge":
        # Recurrent state plus two tensors per layer for an eight-token window.
        return (56 + 3 * 2 * 8 * 56) * FLOAT_BYTES
    if model == "rlt":
        # Encoder output plus per-layer projected global memory and decoder SWA.
        return ((1 + 2 * 2) * context * 40 + 2 * 2 * 8 * 40) * FLOAT_BYTES
    raise KeyError(model)


def make_table(models, contexts):
    return {
        model: {str(context): persistent_state_bytes(model, context) for context in contexts}
        for model in models
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="64,128,256,512,1024,2048,4096")
    parser.add_argument("--out", default="analysis/study2/state_memory.json")
    args = parser.parse_args()
    contexts = [int(x) for x in args.contexts.split(",")]
    models = ["elman", "gru", "lstm", "transformer", "token_merge", "rlt",
              "scar", "scar_carrier", "scar_norecall"]
    table = make_table(models, contexts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"float_bytes": FLOAT_BYTES, "state_bytes": table}, indent=2))
    print(f"wrote persistent-state table to {out}")


if __name__ == "__main__":
    main()
