"""Study 2D selective-copy benchmark.

The input contains marked and unmarked symbol items followed by a copy
decoder.  The decoder emits the marked symbols in their original order.  The
training loss is applied only at copy-slot positions; previous target symbols
are teacher-forced between slots.  Evaluation reports teacher-forced and
free-running accuracy, with free-running metrics used as the primary result.

This module deliberately refuses local execution before constructing an
optimizer.  Training belongs in the Kaggle driver under ``kaggle/selective-copy``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from bench import MODELS, add_bos, count_params


SYMBOL_COUNT = 16
MARK = SYMBOL_COUNT
DISTRACTOR = SYMBOL_COUNT + 1
COPY_START = SYMBOL_COUNT + 2
SLOT = SYMBOL_COUNT + 3
VOCAB = SYMBOL_COUNT + 5
BOS = VOCAB - 1
CHANCE_PCT = 100.0 / SYMBOL_COUNT

DEFAULT_EVAL_MARKED = (1, 2, 4, 8, 16, 32)


def make_batch(
    total_items: int,
    marked_items: int,
    batch_size: int,
    rng: np.random.Generator,
    distractor_vocab: int = SYMBOL_COUNT,
):
    """Create teacher-forced marked-copy examples.

    Every item is encoded as ``[mark-bit, symbol]``.  After the input stream,
    each copy slot is followed by the previous target symbol.  Thus the model
    must retrieve the current marked item at a slot; it cannot see that item's
    answer at the current prediction position.
    """
    if total_items <= 0:
        raise ValueError("total_items must be positive")
    if not 1 <= marked_items <= total_items:
        raise ValueError("marked_items must be in [1, total_items]")
    if not 1 <= distractor_vocab <= SYMBOL_COUNT:
        raise ValueError("distractor_vocab must be in [1, SYMBOL_COUNT]")

    input_len = 2 * total_items + 1
    seq_len = input_len + 2 * marked_items
    seq = np.empty((batch_size, seq_len), dtype=np.int64)
    targets = np.empty((batch_size, marked_items), dtype=np.int64)
    slot_positions = np.arange(
        input_len, input_len + 2 * marked_items, 2, dtype=np.int64
    ) + 1  # +1 for the BOS prefix in model input

    for b in range(batch_size):
        marked = np.sort(rng.choice(total_items, size=marked_items, replace=False))
        symbols = rng.integers(0, distractor_vocab, size=total_items)
        # Marked values use the full alphabet so the task remains nontrivial
        # even when distractor entropy is deliberately reduced.
        symbols[marked] = rng.integers(0, SYMBOL_COUNT, size=marked_items)
        data = np.empty(input_len, dtype=np.int64)
        data[0 : 2 * total_items : 2] = DISTRACTOR
        data[1 : 2 * total_items : 2] = symbols
        data[2 * marked] = MARK
        data[2 * total_items] = COPY_START
        targets[b] = symbols[marked]
        seq[b, :input_len] = data
        for j, value in enumerate(targets[b]):
            seq[b, input_len + 2 * j] = SLOT
            seq[b, input_len + 2 * j + 1] = value
    return (
        torch.from_numpy(seq).long(),
        torch.from_numpy(targets).long(),
        torch.from_numpy(slot_positions).long(),
    )


def model_config(scar_k: int):
    return {
        "elman": {"d": 192},
        "lstm": {"d": 96},
        "gru": {"d": 112},
        "transformer": {"d": 56, "L": 3, "h": 4, "ffn": 112},
        "token_merge": {"d": 56, "L": 3, "h": 4, "ffn": 112, "window": 8},
        "rlt": {"d": 40, "LE": 2, "LD": 2, "h": 4, "ffn": 80, "window": 8},
        "scar": {"d": 88, "k": scar_k, "r": 40},
        "scar_carrier": {
            "d": 112, "k": scar_k, "r": 40,
            "use_memory": False, "use_recall": False,
        },
        "scar_norecall": {
            "d": 98, "k": scar_k, "r": 40,
            "use_memory": True, "use_recall": False,
        },
    }


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _teacher_forced(model, seq, targets, slot_positions, device):
    logits = model(add_bos(seq.to(device), BOS))
    positions = slot_positions.to(device)
    slot_logits = logits[:, positions, :]
    pred = slot_logits.argmax(-1)
    item_acc = (pred == targets.to(device)).float().mean().item() * 100.0
    exact = (pred == targets.to(device)).all(dim=1).float().mean().item() * 100.0
    return item_acc, exact


def _free_decode(model, seq, marked_items, device):
    """Free-running decode after the data stream, batched over examples."""
    input_len = seq.shape[1] - 2 * marked_items
    prefix = seq[:, :input_len].to(device)
    predictions = []
    for _ in range(marked_items):
        slot = torch.full(
            (prefix.shape[0], 1), SLOT, dtype=torch.long, device=device
        )
        prefix = torch.cat([prefix, slot], dim=1)
        logits = model(add_bos(prefix, BOS))[:, -1]
        pred = logits.argmax(-1)
        predictions.append(pred)
        prefix = torch.cat([prefix, pred[:, None]], dim=1)
    return torch.stack(predictions, dim=1)


def evaluate(model, eval_sets, device, eval_batch):
    model.eval()
    free_item, free_exact = {}, {}
    teacher_item, teacher_exact = {}, {}
    elapsed = {}
    with torch.no_grad():
        for marked, (seq, targets, slots) in eval_sets.items():
            n = seq.shape[0]
            free_correct = free_total = free_exact_count = 0
            teacher_correct = teacher_total = teacher_exact_count = 0
            t0 = time.perf_counter()
            for i in range(0, n, eval_batch):
                batch = slice(i, i + eval_batch)
                sb, tb = seq[batch], targets[batch]
                item, exact = _teacher_forced(
                    model, sb, tb, slots, device
                )
                size = sb.shape[0]
                teacher_correct += item * size * marked / 100.0
                teacher_total += size * marked
                teacher_exact_count += exact * size / 100.0

                pred = _free_decode(model, sb, marked, device)
                matches = pred.cpu() == tb
                free_correct += matches.sum().item()
                free_total += matches.numel()
                free_exact_count += matches.all(dim=1).sum().item()
            elapsed[marked] = (time.perf_counter() - t0) / n * 1e3
            free_item[marked] = 100.0 * free_correct / free_total
            free_exact[marked] = 100.0 * free_exact_count / n
            teacher_item[marked] = 100.0 * teacher_correct / teacher_total
            teacher_exact[marked] = 100.0 * teacher_exact_count / n
    model.train()
    return free_item, free_exact, teacher_item, teacher_exact, elapsed


def main():
    if not os.path.isdir("/kaggle/working"):
        raise SystemExit(
            "Refusing local training. Run the selective-copy benchmark through Kaggle."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train_items", type=int, default=16)
    parser.add_argument("--train_marked", type=int, default=4)
    parser.add_argument("--eval_items", type=int, default=64)
    parser.add_argument(
        "--eval_marked", default=",".join(map(str, DEFAULT_EVAL_MARKED))
    )
    parser.add_argument("--eval_examples", type=int, default=4096)
    parser.add_argument("--eval_batch", type=int, default=64)
    parser.add_argument("--distractor_vocab", type=int, default=SYMBOL_COUNT)
    parser.add_argument("--scar_k", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="study2_results/selective_copy")
    parser.add_argument("--study", default="study2")
    parser.add_argument("--protocol_version", default="v3.0")
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    try:
        eval_marked = tuple(
            sorted({int(x) for x in args.eval_marked.split(",") if x.strip()})
        )
    except ValueError as exc:
        raise SystemExit("eval_marked must be comma-separated positive integers") from exc
    if not eval_marked or any(x <= 0 or x > args.eval_items for x in eval_marked):
        raise SystemExit("eval_marked must be in [1, eval_items]")
    if args.train_marked > args.train_items:
        raise SystemExit("train_marked must be <= train_items")
    if args.eval_examples <= 0 or args.eval_batch <= 0 or args.scar_k <= 0:
        raise SystemExit("eval_examples, eval_batch, and scar_k must be positive")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(1000 + args.seed)
    eval_rng = np.random.default_rng(3000 + args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(2)

    cfg = model_config(args.scar_k)
    model = MODELS[args.model](VOCAB, cfg)
    model.to(device)
    params = count_params(model)
    print(f"[{args.model}/selective_copy/s{args.seed}] params={params}", flush=True)

    eval_sets = {}
    for marked in eval_marked:
        eval_sets[marked] = make_batch(
            args.eval_items, marked, args.eval_examples, eval_rng, args.distractor_vocab
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup, total = 200, args.steps

    def lr_fn(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (
            1 + math.cos(math.pi * min((step - warmup) / max(total - warmup, 1), 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    train_rng = np.random.default_rng(2000 + args.seed)
    losses, step_ms = [], []
    training_tokens = 0
    model.train()
    start = time.perf_counter()
    for step in range(args.steps):
        t0 = time.perf_counter()
        seq, targets, slots = make_batch(
            args.train_items,
            args.train_marked,
            args.batch,
            train_rng,
            args.distractor_vocab,
        )
        seq = seq.to(device)
        targets = targets.to(device)
        slots = slots.to(device)
        training_tokens += int(seq.numel())
        logits = model(add_bos(seq, BOS))
        slot_logits = logits[:, slots, :]
        loss = F.cross_entropy(slot_logits.reshape(-1, VOCAB), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        step_ms.append((time.perf_counter() - t0) * 1e3)
        if step % 500 == 0:
            print(f"  step {step} loss {np.mean(losses[-50:]):.4f}", flush=True)

    free_item, free_exact, teacher_item, teacher_exact, eval_ms = evaluate(
        model, eval_sets, device, args.eval_batch
    )
    commit = git_commit()
    eval_tokens = [2 * args.eval_items + 1 + 2 * x for x in eval_marked]
    out = {
        "study": args.study,
        "protocol_version": args.protocol_version,
        "experiment_id": args.experiment_id,
        "git_commit": commit,
        "model": args.model,
        "variant": args.model,
        "task": "selective_copy",
        "seed": args.seed,
        "task_parameters": {
            "train_items": args.train_items,
            "train_marked": args.train_marked,
            "eval_items": args.eval_items,
            "eval_marked": list(eval_marked),
            "distractor_vocab": args.distractor_vocab,
            "symbol_count": SYMBOL_COUNT,
            "free_running_primary": True,
        },
        "params": params,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": 0.01},
        "lr": args.lr,
        "weight_decay": 0.01,
        "warmup_steps": warmup,
        "lr_schedule": "linear_warmup_then_cosine_to_0.1",
        "batch_size": args.batch,
        "steps": args.steps,
        "training_examples": args.steps * args.batch,
        "training_tokens": training_tokens,
        "train_context": 2 * args.train_items + 1 + 2 * args.train_marked,
        "eval_contexts": list(eval_marked),
        "eval_context_tokens": eval_tokens,
        "eval_examples": args.eval_examples,
        "chance_pct": CHANCE_PCT,
        "bos_token": BOS,
        "supervision": "selective_copy_multi_output",
        "curriculum": False,
        "metrics": {"accuracy_pct": {str(k): v for k, v in free_item.items()}},
        "raw_metrics": {
            "free_running_accuracy_pct": {str(k): v for k, v in free_item.items()},
            "free_running_exact_sequence_pct": {str(k): v for k, v in free_exact.items()},
            "teacher_forced_accuracy_pct": {str(k): v for k, v in teacher_item.items()},
            "teacher_forced_exact_sequence_pct": {str(k): v for k, v in teacher_exact.items()},
            "eval_ms_per_program": {str(k): v for k, v in eval_ms.items()},
            "loss_tail": losses[-100:],
            "step_ms_tail": step_ms[-500:],
        },
        "acc": {str(k): v for k, v in free_item.items()},
        "inference_ms_per_example": {str(k): v for k, v in eval_ms.items()},
        "train_ms_per_step": float(np.mean(step_ms[-500:])),
        "final_loss": float(np.mean(losses[-100:])),
        "train_seconds": time.perf_counter() - start,
        "env": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "torch_threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": commit,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.experiment_id}.json"), "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
