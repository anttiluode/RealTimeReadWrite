"""The fact language (PREREG.md, Part 1).

Tokens: PAD FACT IS ASK ANS | 128 entities | 32 values | 200 filler words.
fact      FACT e IS a          (4 tokens)
question  ASK e ANS a          (4 tokens; the model must predict a at the ANS position)
Everything lives in 4-token slots; a chunk is 16 slots = 64 tokens.
"""
import numpy as np

PAD, FACT, IS, ASK, ANS = range(5)
NE, NV, NF = 128, 32, 200
E0 = 5
V0 = E0 + NE
F0 = V0 + NV
VOCAB = F0 + NF
CHUNK, SLOTS = 64, 16


def filler(rng, n=4):
    return list(F0 + rng.integers(0, NF, n))


class Builder:
    """Places 4-token segments into (chunk, slot) positions and records answer targets."""

    def __init__(self, n_chunks, rng):
        self.rng, self.n = rng, n_chunks
        self.seg = [[None] * SLOTS for _ in range(n_chunks)]

    def free(self, c, after=-1):
        return [s for s in range(after + 1, SLOTS) if self.seg[c][s] is None]

    def put(self, c, s, toks, meta=None):
        self.seg[c][s] = (toks, meta)

    def tokens(self):
        toks, tgt, kind, gap = [], [], [], []
        for c in range(self.n):
            for s in range(SLOTS):
                item = self.seg[c][s]
                if item is None:
                    t, m = filler(self.rng), None
                else:
                    t, m = item
                toks += t
                tgt += [-1] * 4
                kind += [""] * 4
                gap += [0] * 4
                if m is not None:          # question: target sits at the ANS position (offset 2)
                    i = len(toks) - 2
                    tgt[i], kind[i], gap[i] = m["value"], m["kind"], m["gap"]
        return np.array(toks), np.array(tgt), np.array(kind), np.array(gap)


def episode(rng, n_chunks=8, n_facts=8, upd_frac=0.3, q_kinds=("cross", "in"), p_cross=0.6, fact_chunks=6):
    """Returns tokens, targets (value index or -1), question kind per position, chunk gap."""
    b = Builder(n_chunks, rng)
    ents = rng.choice(NE, n_facts, replace=False)
    for e in ents:
        a = int(rng.integers(NV))
        cf = int(rng.integers(0, min(fact_chunks, n_chunks)))
        fs = b.free(cf)
        if not fs:
            continue
        sf = int(rng.choice(fs))
        b.put(cf, sf, [FACT, E0 + e, IS, V0 + a])
        last_c, last_s, val, updated = cf, sf, a, False
        if rng.random() < upd_frac and cf + 1 < n_chunks - 1:
            cu = int(rng.integers(cf + 1, n_chunks - 1))
            fu = b.free(cu)
            if fu:
                su = int(rng.choice(fu))
                a2 = int((a + 1 + rng.integers(NV - 1)) % NV)
                b.put(cu, su, [FACT, E0 + e, IS, V0 + a2])
                last_c, last_s, val, updated = cu, su, a2, True
        want_cross = "cross" in q_kinds and (rng.random() < p_cross or "in" not in q_kinds)
        if want_cross and last_c + 1 < n_chunks:
            cq = int(rng.integers(last_c + 1, n_chunks))
            fq = b.free(cq)
            if fq:
                kind = "cross_upd" if updated else "cross"
                b.put(cq, int(rng.choice(fq)), [ASK, E0 + e, ANS, V0 + val], {"value": val, "kind": kind, "gap": cq - last_c})
        elif "in" in q_kinds:
            fq = b.free(last_c, after=last_s)
            if fq:
                b.put(last_c, int(rng.choice(fq)), [ASK, E0 + e, ANS, V0 + val], {"value": val, "kind": "in", "gap": 0})
    return b.tokens()


def batch(rng, B, **kw):
    out = [episode(rng, **kw) for _ in range(B)]
    return tuple(np.stack(z) for z in zip(*out))
