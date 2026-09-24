"""
SCAR research benchmark: state-tracking under controlled supervision density.

Models (9): elman, lstm, gru, transformer (FULL causal attention), token_merge,
rlt, scar, scar_carrier (ablation: memory removed), scar_norecall (ablation:
memory written but never read).
Tasks: parity (chance 50%), five (chance 20%), recall (chance 12.5%); the two
legacy tasks run in dense + sparse supervision, recall is sparse-only.
Protocol: ~75-92K params, 3 seeds, train 32 ops (64 for recall), eval at
16/32/64/128/256 ops (64/128/256/512 for recall), 2048 test programs per length.

Train/eval share ONE input convention: the token sequence is always prefixed
with a BOS token (add_bos). Sparse scoring reads the output at the final
position; dense scoring reads outputs at positions 1..T, aligned with per-step
targets after each input token.
"""
import argparse, json, math, os, platform, subprocess, time
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- tasks ----------------

TASK_CHANCE = {"parity": 50.0, "five": 20.0, "recall": 12.5, "assoc": 6.25}
TASK_VOCAB = {"parity": 5, "five": 11, "recall": 10, "assoc": 50}
EVAL_LENS = {"parity": (16, 32, 64, 128, 256),
             "five": (16, 32, 64, 128, 256),
             "recall": (64, 128, 256, 512),
             "assoc": (1, 2, 4, 8, 16, 32)}
DEFAULT_TRAIN_OPS = {"parity": 32, "five": 32, "recall": 64, "assoc": 4}

def gen_automaton(rng, n=5):
    # random permutation per transition token => T[s][a] gives next state
    return np.stack([rng.permutation(n) for _ in range(n)])

def make_batch(task, ops, bs, rng, auto):
    if task == "parity":
        bits = rng.integers(0, 2, size=(bs, ops))
        ans = bits.sum(axis=1) % 2
        seq = bits
        run = np.cumsum(bits, axis=1) % 2          # per-step targets (dense)
    elif task == "recall":
        # Delayed first-token recall: the answer is seq[0]; the remaining
        # ops-1 symbols are same-vocab distractors. Independent justification:
        # the canonical probe for whether fixed recurrent state retains
        # information over long horizons (copying/recall literature).
        seq = rng.integers(0, 8, size=(bs, ops))
        ans = seq[:, 0].copy()
        run = np.tile(ans[:, None], (1, ops))      # placeholder; recall is sparse-only
    elif task == "assoc":
        # Key/value pairs use disjoint vocabularies. Token 16 is BOS and token
        # 17 is a query marker; keys are 0..7 and values are 8..15. Each key
        # occurs once in the pairs and is queried once at the end, so the
        # answer cannot be recovered from a token-identity shortcut.
        key_count, value_count = 32, 16
        if ops < 1 or ops > key_count:
            raise ValueError("assoc ops must be between 1 and 32 pairs")
        seq = np.empty((bs, 2 * ops + 2), dtype=np.int64)
        ans = np.empty(bs, dtype=np.int64)
        for b in range(bs):
            keys = rng.choice(key_count, size=ops, replace=False)
            values = rng.integers(0, value_count, size=ops)
            query_i = int(rng.integers(0, ops))
            seq[b, 0:2 * ops:2] = keys
            seq[b, 1:2 * ops:2] = values + key_count
            seq[b, 2 * ops] = TASK_VOCAB["assoc"] - 1
            seq[b, 2 * ops + 1] = keys[query_i]
            ans[b] = values[query_i] + key_count
        run = np.tile(ans[:, None], (1, seq.shape[1]))
    else:
        seq = rng.integers(0, 5, size=(bs, ops))
        state = np.zeros(bs, dtype=np.int64)
        states = np.zeros((bs, ops), dtype=np.int64)
        for t in range(ops):
            state = auto[state, seq[:, t]]
            states[:, t] = state
        ans = state
        run = states                                # per-step targets (dense)
    return (torch.from_numpy(seq).long(), torch.from_numpy(ans).long(),
            torch.from_numpy(run).long())

def make_fixed_eval(task, ops, n, rng, auto):
    seqs, anss = [], []
    for _ in range(n):
        s, a, _run = make_batch(task, ops, 1, rng, auto)
        seqs.append(s[0]); anss.append(a[0])
    return torch.stack(seqs), torch.stack(anss)


def add_bos(seq, bos):
    """The single input convention: always prefix BOS. Used by training AND eval."""
    return torch.cat([torch.full((seq.shape[0], 1), bos, dtype=torch.long),
                      seq], dim=1)

# ---------------- modules ----------------

class CausalAttn(nn.Module):
    """Multi-head causal self-attention with optional sliding window."""
    def __init__(self, d, h, window=None):
        super().__init__()
        self.d, self.h, self.w = d, h, window
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        if self.w is not None:
            keep = torch.arange(T, device=x.device)[:, None] >= (torch.arange(T, device=x.device)[None, :] - (self.w - 1))
            causal = causal & keep
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(y)

class Block(nn.Module):
    def __init__(self, d, h, ffn, window=None):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalAttn(d, h, window)
        self.f1, self.f2 = nn.Linear(d, ffn, bias=False), nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.f2(F.gelu(self.f1(self.ln2(x))))

class CrossAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.d, self.h = d, h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

# ---------------- models ----------------

def sinusoidal(T, d, device):
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, d, 2, dtype=torch.float32, device=device)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang[:, : d // 2])
    return pe

class TransformerLM(nn.Module):
    """Standard FULL causal transformer (no sliding window): every position
    attends to all previous positions. Its evaluation lengths exceed the 32-op
    training length, so any long-length degradation reflects positional
    extrapolation / training-length distribution shift, not a truncated window."""
    def __init__(self, vocab, d=56, L=3, h=4, ffn=112, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, ffn) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, targets=None):
        h = self.emb(x) + sinusoidal(x.shape[1], self.emb.embedding_dim, x.device)
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))

class RecurrentCellLM(nn.Module):
    """Elman RNN / GRU / LSTM era baselines: sequential cell, final hidden -> head."""
    def __init__(self, vocab, cell="lstm", d=96, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        if cell == "elman":
            self.cell = nn.RNNCell(d, d)
        elif cell == "gru":
            self.cell = nn.GRUCell(d, d)
        else:
            self.cell = nn.LSTMCell(d, d)
        self.cell_kind = cell
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        d = self.cell.input_size
        h = torch.zeros(B, d, device=x.device)
        c = torch.zeros(B, d, device=x.device) if self.cell_kind == "lstm" else None
        outs = []
        for t in range(T):
            e = self.emb(x[:, t])
            if self.cell_kind == "lstm":
                h, c = self.cell(e, (h, c))
            else:
                h = self.cell(e, h)
            outs.append(h)
        return self.head(self.lnf(torch.stack(outs, dim=1)))

def swa_step_attention(q_x, cache, Wq, Wk, Wv, Wo, h, wsize):
    """One sliding-window self-attention step over a list of cached (k,v) tensors."""
    d = q_x.shape[-1]
    cache.append((Wk(q_x), Wv(q_x)))
    ks = torch.stack([c[0] for c in cache[-wsize:]], dim=1)
    vs = torch.stack([c[1] for c in cache[-wsize:]], dim=1)
    B, D = q_x.shape
    q = Wq(q_x).view(B, h, D // h).unsqueeze(2)          # (B, h, 1, hd)
    k = ks.view(B, -1, h, D // h).transpose(1, 2)         # (B, h, S, hd)
    v = vs.view(B, -1, h, D // h).transpose(1, 2)         # (B, h, S, hd)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(D // h)
    att = att.softmax(-1)
    y = (att @ v).squeeze(2).reshape(B, D)
    return Wo(y)

class TokenMergeLM(nn.Module):
    """Ablation: recurrent decoder merging raw token embedding with state; SWA self-attn only."""
    def __init__(self, vocab, d=56, L=3, h=4, ffn=112, window=8, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.merge = nn.Linear(2 * d, d, bias=False)
        self.h, self.window = h, window
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.wq = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.wv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.wo = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.f1 = nn.ModuleList([nn.Linear(d, ffn, bias=False) for _ in range(L)])
        self.f2 = nn.ModuleList([nn.Linear(ffn, d, bias=False) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        s = torch.zeros(B, self.emb.embedding_dim, device=x.device)
        caches = [[] for _ in self.ln1]
        outs = []
        for t in range(T):
            u = self.merge(torch.cat([self.emb(x[:, t]), s], dim=-1))
            h = u
            for i in range(len(self.ln1)):
                q_x = self.ln1[i](h)
                h = h + swa_step_attention(q_x, caches[i], self.wq[i], self.wk[i], self.wv[i], self.wo[i], self.h, self.window)
                h = h + self.f2[i](F.gelu(self.f1[i](self.ln2[i](h))))
            s = h
            outs.append(s)
        return self.head(self.lnf(torch.stack(outs, dim=1)))

class RLTLite(nn.Module):
    """Faithful small-scale RLT: causal encoder -> global KV memory;
    recurrent decoder with per-layer cross-attention to memory + layerwise SWA cache
    + feedback of final hidden state. Full BPTT."""
    def __init__(self, vocab, d=56, LE=2, LD=2, h=4, ffn=112, window=8, **kw):
        super().__init__()
        self.d, self.h, self.window, self.LD = d, h, window, LD
        self.emb = nn.Embedding(vocab, d)
        self.enc_blocks = nn.ModuleList([Block(d, h, ffn) for _ in range(LE)])
        self.enc_ln = nn.LayerNorm(d)
        self.mk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.mv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.merge = nn.Linear(2 * d, d, bias=False)
        self.ln_cr = nn.ModuleList([nn.LayerNorm(d) for _ in range(LD)])
        self.cq = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.ck = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.cv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.co = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(LD)])
        self.wq = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.wv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.wo = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(LD)])
        self.f1 = nn.ModuleList([nn.Linear(d, ffn, bias=False) for _ in range(LD)])
        self.f2 = nn.ModuleList([nn.Linear(ffn, d, bias=False) for _ in range(LD)])
        self.s0 = nn.Parameter(torch.zeros(d))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        e = self.emb(x)
        for b in self.enc_blocks:
            e = b(e)               # causal encoder, parallel over tokens
        mem_kv = self.enc_ln(e)     # (B,T,d) global encoder memory
        # cross-attn K/V are parameter-only transforms of the encoder memory:
        # precompute for ALL positions once (identical math to per-step projection)
        M_k = [self.ck[i](self.mk[i](mem_kv)) for i in range(self.LD)]
        M_v = [self.cv[i](self.mv[i](mem_kv)) for i in range(self.LD)]
        s = self.s0.unsqueeze(0).expand(B, -1)
        caches = [[] for _ in range(self.LD)]
        outs = []
        for t in range(T):
            u = self.merge(torch.cat([mem_kv[:, t], s], dim=-1))
            h = u
            for i in range(self.LD):
                # cross-attention to global encoder memory M_{<=t}
                q_x = self.ln_cr[i](h)
                q = self.cq[i](q_x).view(B, self.h, self.d // self.h).unsqueeze(2)   # (B,h,1,hd)
                k = M_k[i][:, : t + 1].view(B, t + 1, self.h, -1).transpose(1, 2)
                v = M_v[i][:, : t + 1].view(B, t + 1, self.h, -1).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d // self.h)
                att = att.softmax(-1)
                h = h + self.co[i]((att @ v).squeeze(2).reshape(B, self.d))
                # sliding-window self-attention over decoder activations
                q_x = self.ln1[i](h)
                h = h + swa_step_attention(q_x, caches[i], self.wq[i], self.wk[i], self.wv[i], self.wo[i], self.h, self.window)
                h = h + self.f2[i](F.gelu(self.f1[i](self.ln2[i](h))))
            s = h
            outs.append(s)
        return self.head(self.lnf(torch.stack(outs, dim=1)))


class SCAR(nn.Module):
    """State-Carrier with Attentive Recall.

    O(1) per token, no KV cache, memory never grows: a gated GRU state-carrier
    writes into a multi-timescale compressed memory (k slots, each a learned
    decay-rate EMA), and a small attentive-recall head reads those slots.
    The bet: RLT's benefit comes from state-carriage plus a cheap read path,
    not from O(T) encoder memory or a looped decoder.

    use_memory=False -> pure GRU state-carrier read path (ablation).
    use_recall=False -> memory is written but never read (ablation).
    """
    def __init__(self, vocab, d=88, k=16, r=40, lam_min=0.90, lam_max=0.999,
                 use_memory=True, use_recall=True, **kw):
        super().__init__()
        self.d, self.k, self.r = d, k, r
        self.use_memory, self.use_recall = use_memory, use_recall
        self.emb = nn.Embedding(vocab, d)
        self.cell = nn.GRUCell(d, d)
        # gated write: what the memory absorbs from (token embedding, state)
        if use_memory:
            self.gw = nn.Linear(2 * d, d, bias=True)
            # Per-slot learned decay. IMPORTANT: the parameter is stored in LOGIT
            # space and mapped through sigmoid, so the INITIAL decays are exactly
            # the desired linspace(lam_min, lam_max). The previous version stored
            # log(lam) here and sigmoid'ed it, which collapsed every slot to ~0.5
            # and destroyed the multi-timescale structure.
            self.log_lam = nn.Parameter(logit_init(k, lam_min, lam_max))
        # attentive recall over the k slots
        if use_recall:
            self.ln_r = nn.LayerNorm(d)
            self.wq = nn.Linear(d, r, bias=False)
            self.wk = nn.Linear(d, r, bias=False)
            self.wv = nn.Linear(d, r, bias=False)
            self.wo = nn.Linear(r, d, bias=False)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def decay(self):
        """Current per-slot decay rates in (0, 1); slot i decays towards lam_max
        (slowest timescale). Exposed for tests and analysis."""
        return torch.sigmoid(self.log_lam) if self.use_memory else None

    def forward(self, x, targets=None):
        B, T = x.shape
        h = torch.zeros(B, self.d, device=x.device)
        mem = (torch.zeros(B, self.k, self.d, device=x.device)
               if self.use_memory else None)
        lam = self.decay().view(1, self.k, 1) if self.use_memory else None
        outs = []
        for t in range(T):
            e = self.emb(x[:, t])
            h = self.cell(e, h)
            if self.use_memory:
                u = torch.sigmoid(self.gw(torch.cat([e, h], dim=-1))) * h
                mem = lam * mem + (1 - lam) * u.unsqueeze(1)      # multi-timescale EMA
                read = h
                if self.use_recall:
                    q = self.wq(self.ln_r(h)).view(B, 1, self.r)
                    att = (q @ self.wk(mem).transpose(1, 2)) / math.sqrt(self.r)
                    recall = self.wo((att.softmax(-1) @ self.wv(mem)).squeeze(1))
                    read = h + recall
            else:
                read = h
            outs.append(read)
        return self.head(self.lnf(torch.stack(outs, dim=1)))


MODELS = {
    "elman": lambda v, cfg: RecurrentCellLM(v, cell="elman", **cfg["elman"]),
    "lstm": lambda v, cfg: RecurrentCellLM(v, cell="lstm", **cfg["lstm"]),
    "gru": lambda v, cfg: RecurrentCellLM(v, cell="gru", **cfg["gru"]),
    "transformer": lambda v, cfg: TransformerLM(v, **cfg["transformer"]),
    "token_merge": lambda v, cfg: TokenMergeLM(v, **cfg["token_merge"]),
    "rlt": lambda v, cfg: RLTLite(v, **cfg["rlt"]),
    "scar": lambda v, cfg: SCAR(v, **cfg["scar"]),
    "scar_carrier": lambda v, cfg: SCAR(v, **cfg["scar_carrier"]),
    "scar_norecall": lambda v, cfg: SCAR(v, **cfg["scar_norecall"]),
}

def logit_init(k, lam_min=0.90, lam_max=0.999):
    """Initialize decay parameters in LOGIT space so that after sigmoid the
    decays are exactly linspace(lam_min, lam_max). Slot 0 forgets fastest
    (half-life ~6.6 tokens at 0.90), slot k-1 slowest (half-life ~693 tokens
    at 0.999) -- a genuine two-decade multi-timescale span."""
    lams = torch.linspace(lam_min, lam_max, k)
    return torch.log(lams / (1.0 - lams))



def count_params(m):
    return sum(p.numel() for p in m.parameters())

def evaluate(model, eval_sets, device, bos, eval_batch=256):
    model.eval()
    accs, times = {}, {}
    with torch.no_grad():
        for ops, (seqs, anss) in eval_sets.items():
            seqs, anss = seqs.to(device), anss.to(device)
            n = seqs.shape[0]
            correct = 0
            t0 = time.perf_counter()
            for i in range(0, n, eval_batch):
                xb, ab = seqs[i:i + eval_batch], anss[i:i + eval_batch]
                pred = model(add_bos(xb, bos))[:, -1].argmax(-1)
                correct += (pred == ab).sum().item()
            accs[ops] = 100.0 * correct / n
            times[ops] = (time.perf_counter() - t0) / n * 1e3
    model.train()
    return accs, times


def main():
    if not os.path.isdir("/kaggle/working"):
        raise SystemExit(
            "Refusing local training. Run the benchmark through a Kaggle CPU kernel."
        )
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["parity", "five", "recall", "assoc"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train_ops", type=int, default=None,
                    help="training length; default = 32 (parity/five), 64 (recall)")
    ap.add_argument("--density", choices=["dense", "sparse"], default=None,
                    help="supervision density; default = task legacy (parity dense, five/recall sparse)")
    ap.add_argument("--curriculum", action="store_true",
                    help="sparse supervision only: train lengths grow 4->16->32 over the "
                         "first 60%% of steps, giving the model learnable sparse targets "
                         "before the full 32-bit chain; eval protocol unchanged")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--eval_lengths", default=None,
                    help="comma-separated evaluation lengths; default uses the Study 1 task lengths")
    ap.add_argument("--eval_examples", type=int, default=2048)
    ap.add_argument("--eval_batch", type=int, default=256,
                    help="evaluation batch size; lower this for long full-attention contexts")
    ap.add_argument("--study", default="study1")
    ap.add_argument("--protocol_version", default="study1")
    ap.add_argument("--experiment_id", default=None,
                    help="stable artifact ID; when supplied, output uses <experiment_id>.json")
    args = ap.parse_args()
    if args.train_ops is None:
        args.train_ops = DEFAULT_TRAIN_OPS[args.task]
    if args.density is None:
        args.density = "dense" if args.task == "parity" else "sparse"
    if args.task in {"recall", "assoc"} and args.density != "sparse":
        raise SystemExit(f"{args.task} task is defined sparse-only (one answer after the delay)")
    if args.eval_examples <= 0 or args.eval_batch <= 0:
        raise SystemExit("eval_examples and eval_batch must be positive")
    if args.eval_lengths:
        try:
            eval_lengths = tuple(sorted({int(x) for x in args.eval_lengths.split(",") if x.strip()}))
        except ValueError as exc:
            raise SystemExit("eval_lengths must be comma-separated positive integers") from exc
        if not eval_lengths or any(x <= 0 for x in eval_lengths):
            raise SystemExit("eval_lengths must contain positive integers")
    else:
        eval_lengths = EVAL_LENS[args.task]

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(1000 + args.seed)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        torch.set_num_threads(2)

    vocab = TASK_VOCAB[args.task]
    BOS = vocab - 2
    cfg = {
        "elman": {"d": 192},
        "lstm": {"d": 96},
        "gru": {"d": 112},
        "transformer": {"d": 56, "L": 3, "h": 4, "ffn": 112},
        "token_merge": {"d": 56, "L": 3, "h": 4, "ffn": 112, "window": 8},
        "rlt": {"d": 40, "LE": 2, "LD": 2, "h": 4, "ffn": 80, "window": 8},
        "scar": {"d": 88, "k": 16, "r": 40},
        # ablations: scar_carrier drops memory entirely and widens the carrier
        # (d 88 -> 112) to recover the parameter budget; scar_norecall keeps the
        # full write path but never reads memory, widening d 88 -> 98 for the
        # same reason. Both land within ~1% of scar's parameter count.
        "scar_carrier": {"d": 112, "k": 16, "r": 40, "use_memory": False, "use_recall": False},
        "scar_norecall": {"d": 98, "k": 16, "r": 40, "use_memory": True, "use_recall": False},
    }
    auto = gen_automaton(rng) if args.task == "five" else None
    model = MODELS[args.model](vocab, cfg)
    nparams = count_params(model)
    print(f"[{args.model}/{args.task}/s{args.seed}] params={nparams}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup, total = 200, args.steps
    def lr_fn(s):
        if s < warmup:
            return (s + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - warmup) / max(total - warmup, 1), 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    train_rng = np.random.default_rng(2000 + args.seed)
    eval_rng = np.random.default_rng(3000 + args.seed)
    eval_sets = {}
    for ops in eval_lengths:
        eval_sets[ops] = make_fixed_eval(args.task, ops, args.eval_examples, eval_rng, auto)

    losses, step_ms = [], []
    training_tokens = 0
    model.train()
    t_start = time.perf_counter()
    # curriculum schedule (sparse only): grow the train length 4 -> 8 -> 16 -> 32
    # over the first 60% of steps. Rationale: sparse parity at ops=32 from scratch
    # is a needle-in-haystack optimization problem -- the single end-of-chain
    # gradient signal is too rare to escape the 50/50 plateau within budget
    # (verified: dense parity solves in <500 steps, sparse stalls at ln(2) for
    # 2500 steps). Short chains still carry sparse learnable signal.
    ops_now = args.train_ops
    if args.curriculum:
        sched_ops = [(4, 0.0), (8, 0.2), (16, 0.4), (32, 0.6)]
    for step in range(args.steps):
        t0 = time.perf_counter()
        if args.curriculum:
            frac = step / max(args.steps, 1)
            for ops_c, start in sched_ops:
                if frac >= start:
                    ops_now = ops_c
        seq, ans, run = make_batch(args.task, ops_now, args.batch, train_rng, auto)
        training_tokens += int(seq.numel())
        x = add_bos(seq, BOS)   # same convention as evaluation
        logits = model(x)   # (B, T, vocab)
        if args.density == "dense":
            # dense supervision: predict the target after every input token
            loss = F.cross_entropy(logits[:, 1:].reshape(-1, vocab), run.reshape(-1))
        else:
            # sparse supervision: single answer after the full chain
            loss = F.cross_entropy(logits[:, -1], ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        step_ms.append((time.perf_counter() - t0) * 1e3)
        if step % 500 == 0:
            print(f"  step {step} loss {np.mean(losses[-50:]):.4f}", flush=True)

    accs, eval_times = evaluate(model, eval_sets, device, BOS, eval_batch=args.eval_batch)
    def git_commit():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                text=True).strip()
        except Exception:
            return None

    exp_id = args.experiment_id or f"{args.model}_{args.task}_{args.density}_seed{args.seed}"
    commit = git_commit()
    out = {
        "study": args.study, "protocol_version": args.protocol_version,
        "experiment_id": exp_id,
        "git_commit": commit,
        "model": args.model, "variant": args.model,
        "task": args.task, "seed": args.seed,
        "task_parameters": {
            "density": args.density,
            "train_ops": args.train_ops,
            "curriculum": bool(args.curriculum),
        },
        "params": nparams, "steps": args.steps, "batch": args.batch,
        "batch_size": args.batch,
        "lr": args.lr, "weight_decay": 0.01, "warmup_steps": warmup,
        "lr_schedule": "linear_warmup_then_cosine_to_0.1",
        "train_ops": args.train_ops,
        "train_context": (2 * args.train_ops + 2 if args.task == "assoc" else args.train_ops),
        "training_examples": args.steps * args.batch,
        "training_tokens": training_tokens,
        "curriculum": bool(args.curriculum),
        "eval_lengths": sorted(eval_sets), "eval_examples": args.eval_examples,
        "eval_contexts": sorted(eval_sets),
        "eval_context_tokens": [2 * x + 2 for x in sorted(eval_sets)] if args.task == "assoc" else sorted(eval_sets),
        "chance_pct": TASK_CHANCE[args.task],
        "bos_token": BOS,
        "supervision": f"{args.task}_{args.density}",
        "acc": accs, "metrics": {"accuracy_pct": accs},
        "eval_ms_per_program": eval_times,
        "inference_ms_per_example": eval_times,
        "train_ms_per_step": float(np.mean(step_ms[-500:])),
        "final_loss": float(np.mean(losses[-100:])),
        "train_seconds": time.perf_counter() - t_start,
        "env": {
            "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "device": device, "torch_threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine(),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "git_commit": commit,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": 0.01},
        "raw_metrics": {
            "accuracy_pct": accs,
            "loss_tail": losses[-100:],
            "step_ms_tail": step_ms[-500:],
        },
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{exp_id}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
