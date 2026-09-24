"""Regression tests for the SCAR benchmark fixes.

Run: python3 -m pytest tests/ -v
Covers the three confirmed implementation bugs plus task/model sanity:
1. decay init: sigmoid(logit(lam)) must recover linspace(0.90, 0.999)
2. train/eval BOS consistency: a protocol-aware model must score 100% only
   if evaluation prepends BOS exactly like training
3. transformer: must be full causal attention (and causal, trivially)
4. recall task: answer is the first token; chance is 12.5%
5. ablation param counts: scar_carrier / scar_norecall vs scar
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
import pytest

import bench
from bench import (SCAR, add_bos, make_batch, gen_automaton, make_fixed_eval,
                   evaluate, count_params, MODELS, TASK_CHANCE, logit_init)


# ---------- 1. decay init ----------

def test_decay_init_recovers_intended_range():
    torch.manual_seed(0)
    m = SCAR(vocab=5)
    lam = m.decay()
    # first/last slots land on the intended endpoints, all monotone in between
    assert lam[0].item() == pytest.approx(0.90, abs=1e-6)
    assert lam[-1].item() == pytest.approx(0.999, abs=1e-6)
    assert torch.all(lam[1:] > lam[:-1]), "slots must be monotonically slower"
    assert (lam > 0.85).all() and (lam < 1.0).all()


def test_decay_init_multi_timescale_span():
    # EMA time-constants 1/(1-lam) must span ~2 decades (~10 -> ~1000 tokens)
    lam = torch.sigmoid(logit_init(16))
    taus = (1.0 / (1.0 - lam)).tolist()
    assert 9.0 <= taus[0] <= 11.0
    assert 950.0 <= taus[-1] <= 1050.0
    assert math.log10(taus[-1] / taus[0]) > 1.9


def test_decay_init_does_not_collapse():
    # the old bug: sigmoid(log(lam)) collapsed all slots to ~0.47-0.50 (range ~0.03)
    lam = torch.sigmoid(logit_init(16))
    assert (lam.max() - lam.min()) > 0.08, "decays collapsed -- multi-timescale destroyed"


def test_decay_modes_are_explicit_and_shape_stable():
    for mode in ("learned_multi", "learned_single", "fixed_multi", "fixed_single"):
        m = SCAR(vocab=5, k=8, decay_mode=mode)
        lam = m.decay()
        assert lam.shape == (8,)
        assert torch.all((lam > 0.0) & (lam < 1.0))
    assert torch.allclose(
        SCAR(vocab=5, k=8, decay_mode="learned_single").decay(),
        SCAR(vocab=5, k=8, decay_mode="learned_single").decay()[0].expand(8),
    )


# ---------- 2. train/eval BOS consistency ----------

class ProtocolParityModel(torch.nn.Module):
    """Knows the protocol: input is [BOS, seq...]; answer = parity of x[:, 1:].
    If evaluation forgets to prepend BOS, position 0 holds a real bit and this
    model's answer is shifted by one bit -> ~50% accuracy, failing the test."""

    def __init__(self, vocab=5):
        super().__init__()
        self.vocab = vocab

    def forward(self, x, targets=None):
        bits = (x[:, 1:] % 2 == 1).sum(dim=1) % 2
        logits = torch.full((x.shape[0], x.shape[1], self.vocab), -10.0)
        for b in range(x.shape[0]):
            logits[b, -1, bits[b].item()] = 10.0
        return logits


def test_eval_prepends_bos_and_scores_protocol_consistently():
    model = ProtocolParityModel()
    rng = np.random.default_rng(7)
    seqs, anss = make_fixed_eval("parity", 16, 64, rng, None)
    accs, _ = evaluate(model, {16: (seqs, anss)}, device="cpu", bos=3)
    assert accs[16] == pytest.approx(100.0), (
        "eval protocol broken: model trained with BOS prefix must be evaluated "
        "with the same BOS prefix")


def test_train_uses_bos_prefix():
    # training batches must start with BOS = vocab - 2 for every task
    for task, vocab, chance in [("parity", 5, 50.0), ("five", 11, 20.0),
                                ("recall", 10, 12.5), ("assoc", 50, 6.25)]:
        auto = gen_automaton(np.random.default_rng(3)) if task == "five" else None
        seq, ans, run = make_batch(task, 8, 4, np.random.default_rng(0), auto)
        x = add_bos(seq, vocab - 2)
        expected_len = 2 * 8 + 3 if task == "assoc" else 9
        assert x.shape == (4, expected_len)
        assert (x[:, 0] == vocab - 2).all()
        assert TASK_CHANCE[task] == chance


# ---------- 3. transformer is full causal ----------

def test_transformer_full_causal_attention():
    torch.manual_seed(0)
    m = bench.TransformerLM(vocab=5)
    assert all(bl.attn.w is None for bl in m.blocks), \
        "transformer baseline must be full attention (no sliding window)"
    # causal: changing future tokens must not change earlier outputs
    x = torch.randint(0, 5, (2, 12))
    y1 = m(x)
    x2 = x.clone()
    x2[:, 6:] = (x2[:, 6:] + 1) % 5
    y2 = m(x2)
    assert torch.allclose(y1[:, :6], y2[:, :6], atol=1e-5), "violates causality"
    assert not torch.allclose(y1[:, 6:], y2[:, 6:], atol=1e-5), "future tokens have no effect?"


# ---------- 4. recall task ----------

def test_recall_task_definition():
    seq, ans, run = make_batch("recall", 32, 256, np.random.default_rng(11), None)
    assert seq.shape == (256, 32) and (seq < 8).all() and (seq >= 0).all()
    assert torch.equal(ans, seq[:, 0]), "answer must be the first token"
    assert TASK_CHANCE["recall"] == 12.5
    assert bench.DEFAULT_TRAIN_OPS["recall"] == 64


def test_associative_recall_has_disjoint_keys_values_and_query():
    seq, ans, run = make_batch("assoc", 4, 64, np.random.default_rng(17), None)
    assert seq.shape == (64, 10)  # 4 key/value pairs + query marker/key
    assert (seq[:, :-2] < 48).all()
    assert (seq[:, -2] == 49).all()
    assert (seq[:, -1] < 32).all()
    assert ((ans >= 32) & (ans < 48)).all()
    for row, target in zip(seq.numpy(), ans.numpy()):
        keys = row[:8:2]
        values = row[1:8:2]
        query = row[-1]
        assert (keys == query).sum() == 1
        assert target == values[int(np.flatnonzero(keys == query)[0])]
    assert bench.TASK_CHANCE["assoc"] == 6.25


# ---------- 5. ablation parameter counts ----------

def test_ablation_param_counts_reported_and_close():
    cfg = {"d": 88, "k": 16, "r": 40}
    scar = SCAR(vocab=5, **cfg)
    carrier = SCAR(vocab=5, d=112, use_memory=False, use_recall=False)
    norecall = SCAR(vocab=5, d=98, k=16, r=40, use_recall=False)
    n_scar, n_carrier, n_norecall = (count_params(m) for m in (scar, carrier, norecall))
    # both ablations widen d to close the parameter gap from removing a module
    assert abs(n_carrier - n_scar) / n_scar < 0.03, (n_carrier, n_scar)
    assert abs(n_norecall - n_scar) / n_scar < 0.03, (n_norecall, n_scar)
    print(f"\nscar={n_scar} scar_carrier(d=112)={n_carrier} scar_norecall(d=98)={n_norecall}")


def test_scar_forward_shapes_and_mem_always_alive():
    torch.manual_seed(0)
    m = SCAR(vocab=5)
    out = m(torch.randint(0, 5, (2, 10)))
    assert out.shape == (2, 10, 5)
    for flag in ({"use_memory": False}, {"use_recall": False}):
        m2 = SCAR(vocab=5, **flag)
        assert m2(torch.randint(0, 5, (2, 10))).shape == (2, 10, 5)
