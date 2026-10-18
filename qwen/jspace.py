"""Gate 4 (PREREG_3.md): a native read-write memory in a frozen model's own workspace (J-space).

No hash, no supplied codebook, no trained parameters. Needs a GPU for useful speed.

    pip install torch transformers
    python qwen/jspace.py lens            # J_l for every layer (~15-30 min on an RTX 3060), cached
    python qwen/jspace.py gates           # G4.1-G4.4, writes qwen/results_jspace.json + results/gate4_log.txt

J-lens (Gurnee, Sofroniew, Lindsey et al. 2026, arXiv 2607.15495, section 2.1):
    J_l  = E_{t, t'>=t, prompt} [ d h_final,t' / d h_l,t ]          one d x d matrix per layer
    v_x  = J_l^T (gamma * u_x)                                       the J-lens vector of token x
J-space component of h: nonnegative k-sparse combination of unit J-lens vectors over the whole
vocabulary (k = 16), by matching pursuit with a nonnegative projected-gradient refit.

Memory (G4.4), keys and values both in the J-space:
    write  S <- S + (v - S k) k^T     k = J(h_lr at the token before the value), v = J(h_lw at the value)
    read   o = S k_q                  k_q = J(h_lr at the question's last token)
    inject h_lw[Q] <- h + alpha |h| o/|o|
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

NAME_POOL = """James John Robert Michael William David Richard Joseph Thomas Charles Mary Patricia Jennifer Linda
Elizabeth Barbara Susan Jessica Sarah Karen Daniel Matthew Anthony Mark Paul Steven Andrew Joshua Kevin Brian
George Edward Ronald Timothy Jason Jeffrey Ryan Jacob Gary Nicholas Eric Jonathan Stephen Larry Justin Scott
Brandon Benjamin Samuel Frank Gregory Raymond Alexander Patrick Jack Dennis Jerry Tyler Aaron Henry Peter Adam
Nancy Lisa Betty Margaret Sandra Ashley Kimberly Emily Donna Michelle Dorothy Carol Amanda Melissa Deborah
Stephanie Rebecca Sharon Laura Cynthia Kathleen Amy Angela Helen Anna Emma Olivia Sophia Grace Alice Lucy Oscar
Hugo Leo Max Sam Tom Ben Kate Jane Rose Ruth Vera Eva Nina Lily Mia Zoe Victor Simon Martin Lucas Felix Julia
Claire Diana Paula Monica Teresa Irene Carlos Pedro Juan Maria Hans Karl Otto Ivan Igor Omar Ali Raj Ken""".split()
COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = {"red": "a ripe tomato", "blue": "a clear summer sky", "green": "fresh spring grass", "yellow": "a ripe banana"}
QUESTIONS = ["What color is {n}? Answer:", "Question: What is the color of {n}? Answer:",
             "Give only the color of {n}. Answer:"]
FACT = "The color of {n} is {c}."
YESNO = "Is {n} the same color as {o}? Answer:"
FILLER = ["The weather was mild that afternoon.", "A train left the station on time.",
          "The kitchen smelled of fresh bread.", "Nobody answered the phone.", "The river was higher than usual.",
          "Two birds sat on the old fence.", "The meeting was moved to Tuesday.", "A letter arrived from abroad.",
          "The garden needed water.", "Someone left a window open.", "The shop closed early today.",
          "Snow fell quietly over the town."]
OPENINGS = ["The", "In 1998,", "Scientists have", "Once upon a time", "The recipe calls for", "According to the report,",
            "My grandmother", "The history of", "When the market", "In the morning", "The best way to", "Researchers at",
            "The city council", "It was a", "Most people think", "The new album", "During the war,", "The engine",
            "Water is", "The museum", "Last summer,", "The software", "Many students", "The forest", "A good teacher",
            "The football team", "In mathematics,", "The old house", "Our company", "The river", "Doctors say", "The film"]
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]


# ------------------------------------------------------------------ the frozen model, with hooks
class Frozen:
    """Frozen causal LM. `run` can capture layer outputs and add a delta to any layer's output."""

    def __init__(self, lm, tok, device):
        self.lm, self.tok, self.device = lm, tok, device
        for p in lm.parameters():
            p.requires_grad_(False)
        self.layers = lm.model.layers
        self.L = len(self.layers)
        self.gamma = lm.model.norm.weight.detach().float()
        self.U = lm.get_output_embeddings().weight.detach().float()          # (V, d)
        self.capture, self.cap, self.delta, self.grad_in = set(), {}, {}, False
        lm.model.embed_tokens.register_forward_hook(self._embed)
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_hook(i))
        lm.model.norm.register_forward_pre_hook(self._final)

    def _embed(self, module, args, out):
        return out.requires_grad_(True) if self.grad_in else None

    def _make_hook(self, i):
        def hook(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if i in self.delta:
                h = h + self.delta[i].to(h.dtype)
            if i in self.capture:
                self.cap[i] = h
            if i in self.delta:
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return None
        return hook

    def _final(self, module, args):
        if "final" in self.capture:
            self.cap["final"] = args[0]

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def run(self, ids, capture=(), delta=None):
        """ids: (B, T) or list. Returns logits (B, T, V) and captured {layer: (B, T, d)}."""
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids)
        if ids.dim() == 1:
            ids = ids[None]
        self.capture, self.cap, self.delta = set(capture), {}, dict(delta or {})
        out = self.lm(input_ids=ids.to(self.device), use_cache=False).logits
        cap, self.capture, self.delta = self.cap, set(), {}
        return out, cap


# ------------------------------------------------------------------ the J-lens
def corpus(fz, n_prompts, T, seed=0):
    """Self-sampled text: n_prompts sequences of exactly T tokens, after neutral openings."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    while len(out) < n_prompts:
        ids = torch.tensor([fz.enc(OPENINGS[len(out) % len(OPENINGS)])])
        while ids.shape[1] < T:
            with torch.no_grad():
                logits, _ = fz.run(ids)
            p = torch.softmax(logits[0, -1].float(), -1).cpu()
            ids = torch.cat([ids, torch.multinomial(p, 1, generator=g)[None]], 1)
        out.append(ids[0, :T])
    return torch.stack(out)


def jlens(fz, prompts, layers=None, rows_per_pass=32, log=print):
    """J_l = E_{t, t'>=t, prompt}[d h_final,t' / d h_l,t] for each layer, by batched vector-Jacobian
    products: row i of sum_t sum_{t'} J(t', t) is the gradient of sum_{t'} h_final[t', i] with respect to
    every h_l[t], summed over t (causality makes the t' < t terms zero)."""
    layers = list(range(fz.L)) if layers is None else layers
    d = fz.gamma.shape[0]
    P, T = prompts.shape
    J = {l: torch.zeros(d, d, device=fz.device) for l in layers}
    fz.grad_in = True
    t0 = time.time()
    try:
        for pi in range(P):
            for i0 in range(0, d, rows_per_pass):
                rows = list(range(i0, min(d, i0 + rows_per_pass)))
                ids = prompts[pi][None].repeat(len(rows), 1)
                fz.capture, fz.cap, fz.delta = set(layers) | {"final"}, {}, {}
                fz.lm(input_ids=ids.to(fz.device), use_cache=False)
                cap = fz.cap
                hf = cap["final"]
                G = torch.zeros_like(hf)
                G[torch.arange(len(rows)), :, torch.tensor(rows)] = 1.0
                grads = torch.autograd.grad(hf, [cap[l] for l in layers], grad_outputs=G)
                for l, g in zip(layers, grads):
                    J[l][rows] += g.float().sum(1)
            if (pi + 1) % max(1, P // 10) == 0:
                log(f"  lens: prompt {pi + 1}/{P} ({time.time() - t0:.0f}s)")
    finally:
        fz.grad_in, fz.capture, fz.cap = False, set(), {}
    pairs = P * T * (T + 1) / 2
    return {l: (J[l] / pairs).cpu() for l in layers}


def lens_vectors(fz, J, token_ids):
    """Unit J-lens vectors v_x = J^T (gamma * u_x) for the given token ids: (n, d)."""
    W = fz.gamma[None] * fz.U[token_ids]
    return F.normalize(W @ J.to(W.device), dim=-1)


def logit_vectors(fz, token_ids):
    return F.normalize(fz.gamma[None] * fz.U[token_ids], dim=-1)


class JSpace:
    """J-space component of an activation at one layer: k-sparse nonnegative combination of unit J-lens
    vectors over the whole vocabulary."""

    def __init__(self, fz, J, k=16, chunk=32768):
        self.k = k
        D = []
        for s in range(0, fz.U.shape[0], chunk):        # (V, d), in chunks to bound memory
            W = fz.gamma[None] * fz.U[s:s + chunk]
            D.append(W @ J.to(W.device))
        D = torch.cat(D)
        self.D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def component(self, h, iters=60):
        """h: (d,). Returns (J-space component (d,), atom ids, coefficients)."""
        h = h.float()
        r, sel = h.clone(), []
        coef = torch.zeros(0, device=h.device)
        for _ in range(self.k):
            c = self.D @ r
            if sel:
                c[sel] = -float("inf")
            j = int(c.argmax())
            if c[j] <= 0:
                break
            sel.append(j)
            A = self.D[sel].T                                # (d, s)
            x = torch.cat([coef, c[j:j + 1].clamp_min(0)])
            L = torch.linalg.matrix_norm(A, ord=2) ** 2 + 1e-8
            for _ in range(iters):                           # nonnegative refit on the support
                x = (x - A.T @ (A @ x - h) / L).clamp_min(0)
            coef = x
            r = h - A @ coef
        if not sel:
            return torch.zeros_like(h), [], coef
        return self.D[sel].T @ coef, sel, coef


# ------------------------------------------------------------------ helpers
def single_token_names(fz, n=64):
    ok = [m for m in NAME_POOL if len(fz.enc(" " + m)) == 1]
    seen, out = set(), []
    for m in ok:
        if m not in seen:
            seen.add(m); out.append(m)
    return out[:n]


def split(names):
    dev = [m for i, m in enumerate(names) if i % 4 == 0]
    test = [m for i, m in enumerate(names) if i % 4 != 0]
    return dev, test


def tid(fz, word):
    return fz.enc(" " + word)[0]


def name_pos(fz, text, name):
    ids = fz.enc(text)
    t = tid(fz, name)
    return ids, ids.index(t)


def four(logits_last, cids):
    return int(logits_last[cids].argmax())


# ------------------------------------------------------------------ G4.1 address
@torch.no_grad()
def g41(fz, Js, dev, test, log):
    """One forward per question captures every layer; then score candidates layer by layer."""
    hs = {}
    for group, names in (("dev", dev), ("test", test)):
        for i, m in enumerate(names):
            for q in QUESTIONS:
                ids, sp = name_pos(fz, q.format(n=m), m)
                _, cap = fz.run(ids, capture=range(fz.L))
                hs.setdefault(group, []).append((i, torch.stack([cap[l][0, -1] for l in range(fz.L)]).float(),
                                                 torch.stack([cap[l][0, sp] for l in range(fz.L)]).float()))
    res = {"Q": {}, "S": {}, "Q_logit": {}, "S_logit": {}}
    dev_acc_Q = {}
    Ut = logit_vectors(fz, [tid(fz, m) for m in test])
    for l in range(fz.L):
        Vd = lens_vectors(fz, Js[l], [tid(fz, m) for m in dev])
        Vt = lens_vectors(fz, Js[l], [tid(fz, m) for m in test])
        dev_acc_Q[l] = sum(int((Vd @ F.normalize(hq[l], dim=0)).argmax() == i) for i, hq, _ in hs["dev"]) / len(hs["dev"])
        for key, V, site in (("Q", Vt, 1), ("S", Vt, 2), ("Q_logit", Ut, 1), ("S_logit", Ut, 2)):
            res[key][l] = sum(int((V @ F.normalize(row[site][l], dim=0)).argmax() == row[0]) for row in hs["test"]) / len(hs["test"])
        log(f"  G4.1 layer {l:2d}: dev Q {100 * dev_acc_Q[l]:5.1f}%  test Q {100 * res['Q'][l]:5.1f}%  "
            f"S {100 * res['S'][l]:5.1f}%  | logit-lens Q {100 * res['Q_logit'][l]:5.1f}%  S {100 * res['S_logit'][l]:5.1f}%")
    lr = max(range(fz.L), key=lambda l: (dev_acc_Q[l], -l))
    res["dev_Q"], res["layer_r"] = dev_acc_Q, lr
    res["test_Q_at_lr"] = res["Q"][lr]
    res["pass"] = bool(res["test_Q_at_lr"] >= 0.90)
    return res


# ------------------------------------------------------------------ G4.2 write-back
@torch.no_grad()
def inject_trials(fz, names, layer, alphas, vec_fn, site="Q"):
    """For each name x question x target colour != own answer, inject vec_fn(colour) at `layer`.
    Returns {alpha: (four-colour acc, strict acc, n)}."""
    cids = [tid(fz, c) for c in COLORS]
    acc = {a: [0, 0, 0] for a in alphas}
    for m in names:
        for q in QUESTIONS:
            ids, sp = name_pos(fz, q.format(n=m), m)
            logits, cap = fz.run(ids, capture=[layer])
            own = four(logits[0, -1].float(), cids)
            h = cap[layer][0]
            pos = len(ids) - 1 if site == "Q" else sp
            hn = h[pos].float().norm()
            targets = [j for j in range(len(COLORS)) if j != own]
            rows, meta = [], []
            for a in alphas:
                for j in targets:
                    d = torch.zeros_like(h)
                    d[pos] = (a * hn * vec_fn(j)).to(h.dtype)
                    rows.append(d); meta.append((a, j))
            delta = torch.stack(rows)
            out, _ = fz.run(torch.tensor(ids)[None].repeat(len(rows), 1), delta={layer: delta})
            last = out[:, -1].float()
            for r, (a, j) in enumerate(meta):
                acc[a][0] += int(last[r, cids].argmax() == j)
                acc[a][1] += int(last[r].argmax() == cids[j])
                acc[a][2] += 1
    return {a: (v[0] / v[2], v[1] / v[2], v[2]) for a, v in acc.items()}


@torch.no_grad()
def g42(fz, Js, dev, test, log, seed=0):
    cids = [tid(fz, c) for c in COLORS]
    best, sweep = None, {}
    for l in range(fz.L):
        V = lens_vectors(fz, Js[l], cids)
        r = inject_trials(fz, dev, l, ALPHAS, lambda j: V[j])
        sweep[l] = {a: r[a][0] for a in ALPHAS}
        for a in ALPHAS:
            key = (r[a][0], -a, -l)
            if best is None or key > best[0]:
                best = (key, l, a)
        log(f"  G4.2 dev layer {l:2d}: " + "  ".join(f"a={a:g} {100 * r[a][0]:5.1f}%" for a in ALPHAS))
    _, lw, aw = best
    V = lens_vectors(fz, Js[lw], cids)
    Ulog = logit_vectors(fz, cids)
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = F.normalize(torch.randn(len(COLORS), V.shape[1], generator=g), dim=-1).to(V.device)
    arms = {"J": lambda j: V[j], "logit": lambda j: Ulog[j], "random": lambda j: R[j]}
    res = {"layer_w": lw, "alpha": aw, "dev_sweep": sweep, "test": {}}
    for arm, fn in arms.items():
        a4, st, n = inject_trials(fz, test, lw, [aw], fn)[aw]
        res["test"][arm] = {"four": a4, "strict": st, "n": n}
    a4, st, n = inject_trials(fz, test, lw, [0.0], lambda j: V[j])[0.0]
    res["test"]["none"] = {"four": a4, "strict": st, "n": n}
    # site S (the name token), reported only: its own dev sweep for alpha at layer lw
    rs = inject_trials(fz, dev, lw, ALPHAS, lambda j: V[j], site="S")
    a_s = max(ALPHAS, key=lambda a: (rs[a][0], -a))
    a4, st, n = inject_trials(fz, test, lw, [a_s], lambda j: V[j], site="S")[a_s]
    res["test"]["J_siteS"] = {"four": a4, "strict": st, "n": n, "alpha": a_s}
    res["pass"] = bool(res["test"]["J"]["four"] >= 0.90 and res["test"]["random"]["four"] <= 0.40)
    return res


# ------------------------------------------------------------------ G4.3 generalization
@torch.no_grad()
def g43(fz, Js, test, lw, aw, log, seed=0):
    rng = random.Random(seed)
    cids = [tid(fz, c) for c in COLORS]
    yes, no = fz.enc(" Yes")[0], fz.enc(" No")[0]
    V = lens_vectors(fz, Js[lw], cids)
    Ulog = logit_vectors(fz, cids)
    items = []
    for i, m in enumerate(test):
        c = rng.randrange(4)
        match = i % 2 == 0
        o = c if match else rng.choice([j for j in range(4) if j != c])
        items.append((m, c, o, match))
    score = {k: [0, 0, 0, 0] for k in ("ceiling", "J", "logit", "none")}   # tp, n_pos, tn, n_neg

    def tally(k, said_yes, match):
        s = score[k]
        if match:
            s[0] += int(said_yes); s[1] += 1
        else:
            s[2] += int(not said_yes); s[3] += 1

    for m, c, o, match in items:
        q = YESNO.format(n=m, o=OBJECTS[COLORS[o]])
        ctx = fz.enc(FACT.format(n=m, c=COLORS[c]) + " " + q)
        out, _ = fz.run(ctx)
        tally("ceiling", bool(out[0, -1, yes] > out[0, -1, no]), match)
        ids = fz.enc(q)
        out, cap = fz.run(ids, capture=[lw])
        tally("none", bool(out[0, -1, yes] > out[0, -1, no]), match)
        h = cap[lw][0]
        hn = h[-1].float().norm()
        for k, vec in (("J", V[c]), ("logit", Ulog[c])):
            d = torch.zeros_like(h)
            d[-1] = (aw * hn * vec).to(h.dtype)
            out, _ = fz.run(ids, delta={lw: d[None]})
            tally(k, bool(out[0, -1, yes] > out[0, -1, no]), match)
    res = {k: 0.5 * (s[0] / max(s[1], 1) + s[2] / max(s[3], 1)) for k, s in score.items()}
    res["n"] = len(items)
    res["informative"] = bool(res["ceiling"] >= 0.75)
    res["pass"] = bool(res["informative"] and res["J"] >= 0.75)
    res["prediction_J_beats_logit_by_10"] = bool(res["J"] >= res["logit"] + 0.10)
    return res


# ------------------------------------------------------------------ G4.4 live native memory
class NativeMemory:
    """S in R^{d x d}; keys/values are J-space components; delta-rule writes."""

    def __init__(self, d, device):
        self.S = torch.zeros(d, d, device=device)

    def write(self, k, v):
        self.S += torch.outer(v - self.S @ k, k)

    def read(self, k):
        return self.S @ k


class Live:
    def __init__(self, fz, jr, jw, lr, lw, alpha, m_w, m_q):
        self.fz, self.jr, self.jw, self.lr, self.lw, self.alpha = fz, jr, jw, lr, lw, alpha
        self.m_w, self.m_q = m_w, m_q

    def key(self, h, mean, center=True):
        c = self.jr.component(h)[0]
        return F.normalize(c - mean if center else c, dim=0)

    def fact_kv(self, name, color):
        """Key at the token before the value, value at the value token (template positions only).
        Returns (centred key, raw key, value)."""
        ids = self.fz.enc(FACT.format(n=name, c=color))
        vpos = ids.index(tid(self.fz, color))
        _, cap = self.fz.run(ids, capture=[self.lr, self.lw])
        c = self.jr.component(cap[self.lr][0, vpos - 1].float())[0]
        v = F.normalize(self.jw.component(cap[self.lw][0, vpos].float())[0], dim=0)
        return F.normalize(c - self.m_w, dim=0), F.normalize(c, dim=0), v

    def q_key(self, ids):
        """(centred key, raw key) at the question's last token."""
        _, cap = self.fz.run(ids, capture=[self.lr])
        c = self.jr.component(cap[self.lr][0, -1].float())[0]
        return F.normalize(c - self.m_q, dim=0), F.normalize(c, dim=0)

    def answer(self, ids, o):
        """Second pass: inject alpha |h| o/|o| at the last position of layer lw."""
        if o is None or o.norm() < 1e-8:
            out, _ = self.fz.run(ids)
            return out[0, -1].float()
        _, cap = self.fz.run(ids, capture=[self.lw])
        h = cap[self.lw][0]
        d = torch.zeros_like(h)
        d[-1] = (self.alpha * h[-1].float().norm() * o / o.norm()).to(h.dtype)
        out, _ = self.fz.run(ids, delta={self.lw: d[None]})
        return out[0, -1].float()


@torch.no_grad()
def calibrate(fz, jr, lr, dev, seed=0):
    rng = random.Random(seed)
    kw, kq = [], []
    for i in range(64):
        m, c = dev[i % len(dev)], COLORS[rng.randrange(4)]
        ids = fz.enc(FACT.format(n=m, c=c))
        vpos = ids.index(tid(fz, c))
        _, cap = fz.run(ids, capture=[lr])
        kw.append(jr.component(cap[lr][0, vpos - 1].float())[0])
    for m in dev:
        for q in QUESTIONS:
            _, cap = fz.run(fz.enc(q.format(n=m)), capture=[lr])
            kq.append(jr.component(cap[lr][0, -1].float())[0])
    return torch.stack(kw).mean(0), torch.stack(kq).mean(0)


@torch.no_grad()
def g44(fz, live, test, Js, log, sizes=(8, 16, 32), episodes=20, seed=0):
    cids = [tid(fz, c) for c in COLORS]
    oracle_V = {m: lens_vectors(fz, Js[live.lr], [tid(fz, m)])[0] for m in test}
    res = {}
    for N in sizes:
        rng = random.Random(seed + N)
        acc = {k: 0 for k in ("L", "stale", "none", "ctx", "L_raw", "oracle_key", "wrong_address_hit")}
        n = 0
        for ep in range(episodes):
            names = rng.sample(test, min(N, len(test)))
            first = {m: rng.randrange(4) for m in names}
            latest = {m: rng.choice([j for j in range(4) if j != first[m]]) for m in names}
            stream1 = [("fact", m, first[m]) for m in names]
            rng.shuffle(stream1)
            stream2 = [("fact", m, latest[m]) for m in names] + [("filler", rng.choice(FILLER), None) for _ in range(2 * N)]
            rng.shuffle(stream2)
            mems = {k: NativeMemory(fz.gamma.shape[0], fz.device) for k in ("L", "L_raw", "oracle_key")}
            stale = None
            for phase, stream in ((1, stream1), (2, stream2)):
                for kind, m, c in stream:
                    if kind != "fact":
                        continue                     # filler is read but writes nothing at the template positions
                    k, kr, v = live.fact_kv(m, COLORS[c])
                    mems["L"].write(k, v)
                    mems["L_raw"].write(kr, v)
                    mems["oracle_key"].write(oracle_V[m], v)
                if phase == 1:
                    stale = NativeMemory(fz.gamma.shape[0], fz.device); stale.S = mems["L"].S.clone()
            sentences = [FACT.format(n=m, c=COLORS[c]) if kind == "fact" else m for kind, m, c in stream1 + stream2]
            for m in names:
                q = rng.choice(QUESTIONS).format(n=m)
                ids = fz.enc(q)
                kq, kq_raw = live.q_key(ids)
                other = rng.choice([x for x in names if x != m]) if len(names) > 1 else m
                k_other = live.q_key(fz.enc(QUESTIONS[0].format(n=other)))[0]
                tgt = latest[m]
                acc["L"] += int(four(live.answer(ids, mems["L"].read(kq)), cids) == tgt)
                acc["stale"] += int(four(live.answer(ids, stale.read(kq)), cids) == tgt)
                acc["none"] += int(four(live.answer(ids, None), cids) == tgt)
                acc["L_raw"] += int(four(live.answer(ids, mems["L_raw"].read(kq_raw)), cids) == tgt)
                acc["oracle_key"] += int(four(live.answer(ids, mems["oracle_key"].read(oracle_V[m])), cids) == tgt)
                acc["wrong_address_hit"] += int(four(live.answer(ids, mems["L"].read(k_other)), cids) == tgt)
                ctx = fz.enc(" ".join(sentences) + " " + q)
                out, _ = fz.run(ctx)
                acc["ctx"] += int(four(out[0, -1].float(), cids) == tgt)
                n += 1
        res[N] = {k: v / n for k, v in acc.items()} | {"n": n}
        log(f"  G4.4 N={N:2d}: " + "  ".join(f"{k} {100 * v:5.1f}%" for k, v in res[N].items() if k != "n"))
    s, b = min(sizes), max(sizes)
    res["pass"] = bool(res[s]["L"] >= 0.80 and res[b]["L"] >= 0.60 and res[s]["L"] - res[s]["none"] >= 0.40)
    return res


# ------------------------------------------------------------------ commands
def load(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True       # fp32 weights, TF32 matmuls (Ampere and later)
        torch.backends.cudnn.allow_tf32 = True
    return tok, lm


def cache_path(args):
    return os.path.join(HERE, "cache", f"jlens_{args.model.replace('/', '_')}_{args.n_prompts}x{args.T}.pt")


def cmd_lens(args, fz=None, log=print):
    if fz is None:
        tok, lm = load(args)
        fz = Frozen(lm, tok, args.device)
    log(f"lens: sampling {args.n_prompts} prompts of {args.T} tokens")
    prompts = corpus(fz, args.n_prompts, args.T, seed=0)
    Js = jlens(fz, prompts, rows_per_pass=args.rows, log=log)
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    torch.save({"J": Js, "prompts": prompts}, cache_path(args))
    log(f"lens: saved {cache_path(args)}")
    return Js


def cmd_gates(args, fz=None):
    os.makedirs(RES, exist_ok=True)
    logf = open(os.path.join(RES, "gate4_log.txt"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    if fz is None:
        tok, lm = load(args)
        fz = Frozen(lm, tok, args.device)
    log(f"=== Gate 4 (PREREG_3) {time.strftime('%Y-%m-%d %H:%M')} model {args.model}")
    Js = torch.load(cache_path(args))["J"] if os.path.exists(cache_path(args)) else cmd_lens(args, fz, log)
    names = single_token_names(fz)
    dev, test = split(names)
    log(f"names: {len(names)} single-token ({len(dev)} dev, {len(test)} test); test: {' '.join(test)}")
    R = {"names": {"dev": dev, "test": test}}
    t0 = time.time()
    R["G4.1"] = g41(fz, Js, dev, test, log)
    lr = R["G4.1"]["layer_r"]
    log(f"G4.1 read layer {lr}: test Q {100 * R['G4.1']['test_Q_at_lr']:.1f}% -> {'PASS' if R['G4.1']['pass'] else 'FAIL'}  ({time.time() - t0:.0f}s)")
    R["G4.2"] = g42(fz, Js, dev, test, log)
    lw, aw = R["G4.2"]["layer_w"], R["G4.2"]["alpha"]
    log(f"G4.2 write layer {lw}, alpha {aw}: " + "  ".join(f"{k} {100 * v['four']:.1f}%" for k, v in R["G4.2"]["test"].items())
        + f" -> {'PASS' if R['G4.2']['pass'] else 'FAIL'}  ({time.time() - t0:.0f}s)")
    R["G4.3"] = g43(fz, Js, test, lw, aw, log)
    g3 = R["G4.3"]
    log(f"G4.3 balanced acc: ceiling {100 * g3['ceiling']:.1f}%  J {100 * g3['J']:.1f}%  logit {100 * g3['logit']:.1f}%  "
        f"none {100 * g3['none']:.1f}% -> {'PASS' if g3['pass'] else ('FAIL' if g3['informative'] else 'UNINFORMATIVE')}")
    jr = JSpace(fz, Js[lr], k=args.k)
    jw = jr if lw == lr else JSpace(fz, Js[lw], k=args.k)
    m_w, m_q = calibrate(fz, jr, lr, dev)
    live = Live(fz, jr, jw, lr, lw, aw, m_w, m_q)
    R["G4.4"] = g44(fz, live, test, Js, log, sizes=tuple(args.sizes), episodes=args.episodes)
    log(f"G4.4 -> {'PASS' if R['G4.4']['pass'] else 'FAIL'}  ({time.time() - t0:.0f}s)")
    R["pass"] = {g: R[g]["pass"] for g in ("G4.1", "G4.2", "G4.3", "G4.4")}
    log(f"pass: {R['pass']}")
    json.dump(R, open(os.path.join(HERE, "results_jspace.json"), "w"), indent=1, default=str)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lens", "gates"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--rows", type=int, default=32, help="Jacobian rows per backward pass (lower if out of memory)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "lens":
        cmd_lens(a)
    else:
        cmd_gates(a)
