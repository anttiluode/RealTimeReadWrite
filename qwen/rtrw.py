"""RealTimeReadWrite on a real frozen LLM (PREREG.md Part 2). Needs a GPU for useful speed.

    pip install torch transformers
    python qwen/rtrw.py train --model Qwen/Qwen2.5-0.5B --steps 2000
    python qwen/rtrw.py gates --model Qwen/Qwen2.5-0.5B

The model's own weights never change. Text is fed in chunks; each chunk starts with an empty KV
cache, so the model cannot attend to anything before the chunk. A fact stated in an earlier chunk
can only reach a later question through the memory, which is read and written on every token:

    read ŷ = S k ;  residue r = v − ŷ ;  write S ← αS + β r kᵀ ;  answer o = S q  (added before the final norm)
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ the memory
class ReadWriteMemory(nn.Module):
    def __init__(self, d_model, dk=64, rule="delta"):
        super().__init__()
        self.rule, self.dk = rule, dk
        self.k = nn.Linear(d_model, dk, bias=False)
        self.v = nn.Linear(d_model, dk, bias=False)
        self.q = nn.Linear(d_model, dk, bias=False)
        self.beta = nn.Linear(d_model, 1)
        self.alpha = nn.Linear(d_model, 1)
        nn.init.zeros_(self.alpha.weight); nn.init.constant_(self.alpha.bias, 4.0)   # start nearly non-forgetting
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)
        self.out = nn.Linear(dk, d_model, bias=False)
        nn.init.normal_(self.out.weight, std=0.02)
        self.S = None

    def reset(self, batch, device):
        self.S = torch.zeros(batch, self.dk, self.dk, device=device)

    def forward(self, h):
        """h: (B, T, d) hidden states of one chunk. Reads and writes S on every token, in order."""
        h = h.float()
        k = F.normalize(self.k(h), dim=-1)
        q = F.normalize(self.q(h), dim=-1)
        v = self.v(h)
        b = torch.sigmoid(self.beta(h))[..., 0]
        a = torch.sigmoid(self.alpha(h))[..., 0]
        S, outs = self.S, []
        for t in range(h.shape[1]):
            kt = k[:, t]
            pred = torch.einsum("bij,bj->bi", S, kt)                     # read
            w = v[:, t] - pred if self.rule == "delta" else v[:, t]      # residue (additive: value)
            S = a[:, t, None, None] * S + b[:, t, None, None] * torch.einsum("bi,bj->bij", w, kt)   # write
            outs.append(torch.einsum("bij,bj->bi", S, q[:, t]))         # answer
        self.S = S
        return self.out(torch.stack(outs, 1))


class LiveModel:
    """Frozen causal LM + memory: read at layer `layer`, inject before the final norm."""

    def __init__(self, lm, layer, mem):
        self.lm, self.mem, self.inj = lm, mem, None
        for p_ in lm.parameters():
            p_.requires_grad_(False)
        layers = lm.model.layers
        layers[layer].register_forward_hook(self._read)
        lm.model.norm.register_forward_pre_hook(self._inject)

    def _read(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        self.inj = self.mem(h.detach()) if self.mem is not None else None

    def _inject(self, module, args):
        if self.inj is None:
            return None
        return (args[0] + self.inj.to(args[0].dtype),) + tuple(args[1:])

    def chunk_logits(self, ids):
        """ids (B, T): one chunk, empty KV cache. Returns logits (B, T, V)."""
        return self.lm(input_ids=ids, use_cache=False).logits


# ------------------------------------------------------------------ the fact text
FIRST = ["Mira", "Oskar", "Lena", "Tomas", "Aino", "Rafael", "Sanna", "Viktor", "Elin", "Jonas", "Ilse", "Pavel",
         "Noora", "Kasper", "Hedda", "Milan", "Saga", "Teodor", "Anja", "Emil", "Ruth", "Felix", "Iris", "Bruno"]
LAST = ["Halden", "Korpela", "Varga", "Lindqvist", "Moreau", "Tanaka", "Okafor", "Brandt", "Salo", "Novak",
        "Ferreira", "Holm", "Kowalski", "Reyes", "Aalto", "Lund", "Petrov", "Nakamura", "Rossi", "Berg"]
ANIMALS = ["tiger", "horse", "eagle", "whale", "rabbit", "wolf", "snake", "otter", "camel", "goat", "bear", "fox",
           "deer", "duck", "frog", "owl", "shark", "mouse", "sheep", "lion", "monkey", "turtle", "crab", "bee",
           "zebra", "goose", "pig", "cow", "cat", "dog", "bat", "ant"]
ADJ = ["quiet", "old", "small", "bright", "cold", "narrow", "long", "green", "heavy", "empty", "soft", "dark"]
NOUN = ["road", "table", "window", "river", "garden", "station", "bridge", "kitchen", "forest", "letter", "market", "tower"]
VERB = ["stood", "waited", "rested", "leaned", "appeared", "remained", "shone", "turned"]


def single_token_animals(tok):
    ok = [a for a in ANIMALS if len(tok.encode(" " + a, add_special_tokens=False)) == 1]
    return ok


class Episodes:
    def __init__(self, tok, chunk=128, seed=0):
        self.tok, self.chunk, self.rng = tok, chunk, random.Random(seed)
        self.animals = single_token_animals(tok)
        self.animal_ids = [tok.encode(" " + a, add_special_tokens=False)[0] for a in self.animals]
        self.enc = lambda s: tok.encode(s, add_special_tokens=False)

    def filler(self):
        r = self.rng
        return f" The {r.choice(ADJ)} {r.choice(NOUN)} {r.choice(VERB)} near the {r.choice(ADJ)} {r.choice(NOUN)}."

    def make(self, n_chunks=4, n_facts=8, upd_frac=0.3, kinds=("cross", "in")):
        """Returns ids (n_chunks*chunk,), targets (index into animals or -1), kind per position.
        A person's sentences are placed only if every chunk they touch still has room, so no question
        is ever separated from its fact by truncation."""
        r = self.rng
        people = r.sample([(f, l) for f in FIRST for l in LAST], n_facts)
        plan = [[] for _ in range(n_chunks)]
        room = [self.chunk - 2] * n_chunks
        L = lambda text, ans=False: len(self.enc(text)) + (1 if ans else 0)
        for f, l in people:
            name = f"{f} {l}"
            a = r.randrange(len(self.animals))
            cf = r.randrange(0, max(1, n_chunks - 1))
            adds = [(cf, f" The favourite animal of {name} is the {self.animals[a]}.", None)]
            last, val, upd = cf, a, False
            if r.random() < upd_frac and cf + 1 < n_chunks - 1:
                cu = r.randrange(cf + 1, n_chunks - 1)
                a2 = (a + 1 + r.randrange(len(self.animals) - 1)) % len(self.animals)
                adds.append((cu, f" Correction: the favourite animal of {name} is now the {self.animals[a2]}.", None))
                last, val, upd = cu, a2, True
            q = f" Question: what is the favourite animal of {name}? Answer: the"
            if "cross" in kinds and last + 1 < n_chunks and (r.random() < 0.6 or "in" not in kinds):
                adds.append((r.randrange(last + 1, n_chunks), q, (val, "cross_upd" if upd else "cross")))
            elif "in" in kinds:
                adds.append((last, q, (val, "in")))
            need = [0] * n_chunks
            for c, text, meta in adds:
                need[c] += L(text, meta is not None)
            if all(need[c] <= room[c] for c in range(n_chunks)):
                for c, text, meta in adds:
                    plan[c].append((text, meta)); room[c] -= L(text, meta is not None)
        ids, tgt, kind = [], [], []
        for c in range(n_chunks):
            order = [s for s in plan[c] if s[1] is None] + [s for s in plan[c] if s[1] is not None]
            cids, ctgt, ckind = [], [], []
            for text, meta in order:
                e = self.enc(text) + ([self.animal_ids[meta[0]]] if meta is not None else [])
                cids += e; ctgt += [-1] * len(e); ckind += [""] * len(e)
                if meta is not None:
                    ctgt[-2], ckind[-2] = meta[0], meta[1]   # predict the animal from the token before it
            while True:                                       # pad with whole filler sentences
                e = self.enc(self.filler())
                if len(cids) + len(e) > self.chunk:
                    break
                cids += e; ctgt += [-1] * len(e); ckind += [""] * len(e)
            pad = self.chunk - len(cids)
            pid = self.enc(" .")[0]
            cids += [pid] * pad; ctgt += [-1] * pad; ckind += [""] * pad
            ids += cids; tgt += ctgt; kind += ckind
        return ids, tgt, kind

    def batch(self, B, **kw):
        out = [self.make(**kw) for _ in range(B)]
        return (torch.tensor([o[0] for o in out]), torch.tensor([o[1] for o in out]), [o[2] for o in out])


def run_episode(live, ids, chunk):
    """Feed chunks in order with a fresh KV cache each time; memory S carries across. Returns logits."""
    B = ids.shape[0]
    if live.mem is not None:
        live.mem.reset(B, ids.device)
    return torch.cat([live.chunk_logits(ids[:, s:s + chunk]) for s in range(0, ids.shape[1], chunk)], 1)


def answer_loss(logits, tgt, animal_ids):
    sub = logits[..., animal_ids].float()
    m = tgt >= 0
    return F.cross_entropy(sub[m], tgt[m]) if m.any() else sub.sum() * 0


# ------------------------------------------------------------------ commands
def load(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    return tok, lm


def train(args, tok=None, lm=None):
    tok, lm = (tok, lm) if lm is not None else load(args)
    d = lm.config.hidden_size
    layer = args.layer if args.layer >= 0 else lm.config.num_hidden_layers // 2
    mem = ReadWriteMemory(d, args.dk, args.rule).to(args.device)
    live = LiveModel(lm, layer, mem)
    ep = Episodes(tok, args.chunk, seed=1)
    aid = torch.tensor(ep.animal_ids, device=args.device)
    opt = torch.optim.AdamW(mem.parameters(), lr=args.lr, weight_decay=0.0)
    t0 = time.time()
    for i in range(args.steps):
        ids, tgt, _ = ep.batch(args.batch, n_chunks=args.chunks, n_facts=random.randint(2, 16), upd_frac=0.3)
        logits = run_episode(live, ids.to(args.device), args.chunk)
        loss = answer_loss(logits, tgt.to(args.device), aid)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % max(1, args.steps // 20) == 0:
            print(f"[{args.rule}] step {i + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    torch.save({"state": mem.state_dict(), "layer": layer, "dk": args.dk, "rule": args.rule},
               os.path.join(HERE, "cache", f"mem_{args.rule}.pt"))
    return mem


@torch.no_grad()
def accuracy(live, ep, aid, n, device, chunk, want, **kw):
    hit = cnt = 0
    for _ in range(n // 8):
        ids, tgt, kind = ep.batch(8, **kw)
        logits = run_episode(live, ids.to(device), chunk)
        pred = logits[..., aid].argmax(-1).cpu()
        m = (tgt >= 0) & torch.tensor([[k in want for k in row] for row in kind])
        hit += int((pred[m] == tgt[m]).sum()); cnt += int(m.sum())
    return hit / max(cnt, 1), cnt


def gates(args):
    tok, lm = load(args)
    d = lm.config.hidden_size
    res = {}
    arms = {"B0": None, "B1": "additive", "M": "delta"}
    lives = {}
    for arm, rule in arms.items():
        if rule is None:
            lives[arm] = (None, None)
            continue
        ck = torch.load(os.path.join(HERE, "cache", f"mem_{rule}.pt"), map_location=args.device)
        mem = ReadWriteMemory(d, ck["dk"], rule).to(args.device)
        mem.load_state_dict(ck["state"])
        lives[arm] = (mem, ck["layer"])
    layer = next(l for m, l in lives.values() if l is not None)
    live = LiveModel(lm, layer, None)                # one set of hooks; swap the memory per arm
    conds = {"R1": (dict(n_facts=8, upd_frac=0.0, kinds=("cross",)), {"cross"}),
             "R2": (dict(n_facts=16, upd_frac=0.0, kinds=("cross",)), {"cross"}),
             "R3": (dict(n_facts=8, upd_frac=1.0, kinds=("cross",)), {"cross_upd"}),
             "R4": (dict(n_facts=8, upd_frac=0.0, kinds=("in",)), {"in"})}
    for g, (kw, want) in conds.items():
        res[g] = {}
        for arm in arms:
            live.mem = lives[arm][0]
            ep = Episodes(tok, args.chunk, seed=1000 + int(g[1]))
            aid = torch.tensor(ep.animal_ids, device=args.device)
            acc, n = accuracy(live, ep, aid, args.n_eval, args.device, args.chunk, want, n_chunks=args.chunks, **kw)
            res[g][arm] = acc
        print(g, {a: f"{100 * v:.1f}%" for a, v in res[g].items()}, flush=True)
    res["pass"] = {"R1": res["R1"]["M"] >= 0.80 and res["R1"]["B0"] <= 0.10,
                   "R2": res["R2"]["M"] >= res["R2"]["B1"] + 0.10,
                   "R3": res["R3"]["M"] >= 0.70 and res["R3"]["M"] >= res["R3"]["B1"] + 0.20,
                   "R4": res["R4"]["M"] >= res["R4"]["B0"] - 0.05}
    print(res["pass"])
    json.dump(res, open(os.path.join(HERE, "results_qwen.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "gates"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rule", default="delta", choices=["delta", "additive"])
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--dk", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n_eval", type=int, default=400)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        gates(a)
