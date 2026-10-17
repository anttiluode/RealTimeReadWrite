"""Pre-registered gates R1-R5 (PREREG.md Part 1). python toy/gates.py  (after toy/train.py)"""
import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from task import batch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(os.path.dirname(HERE), "results")
WIN = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 1
TAG = "" if WIN == 1 else f"_w{WIN}"
LOG = open(os.path.join(RES, f"gate_log{TAG}.txt"), "a")


def log(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def load(name):
    return jax.tree_util.tree_map(jnp.asarray, pickle.load(open(os.path.join(CACHE, name), "rb")))


p = load("base.pkl")
mems = {"delta": load(f"mem_delta{TAG}.pkl"), "additive": load(f"mem_additive{TAG}.pkl")}
f = {None: jax.jit(lambda x: M.forward(p, None, x)),
     "delta": jax.jit(lambda x: M.forward(p, mems["delta"], x, "delta")),
     "additive": jax.jit(lambda x: M.forward(p, mems["additive"], x, "additive"))}
ARMS = {"B0": None, "B1": "additive", "M": "delta"}


def evaluate(seed, n=1000, kinds=None, **kw):
    """Accuracy per arm on targets whose kind is in `kinds`; also accuracy by chunk gap."""
    rng = np.random.default_rng(seed)
    hit = {a: 0 for a in ARMS}
    cnt = 0
    by_gap = {a: {} for a in ARMS}
    for _ in range(n // 50):
        toks, tgt, kind, gap = batch(rng, 50, **kw)
        sel = (tgt >= 0) & np.isin(kind, kinds)
        cnt += sel.sum()
        for a, rule in ARMS.items():
            pred = np.argmax(np.asarray(f[rule](jnp.asarray(toks))), -1)
            ok = (pred == tgt) & sel
            hit[a] += ok.sum()
            for g in np.unique(gap[sel]):
                s = sel & (gap == g)
                h, c = by_gap[a].get(int(g), (0, 0))
                by_gap[a][int(g)] = (h + int(ok[s].sum()), c + int(s.sum()))
    return {a: hit[a] / max(cnt, 1) for a in ARMS} | {"n_questions": int(cnt),
            "by_gap": {a: {g: h / c for g, (h, c) in sorted(v.items())} for a, v in by_gap.items()}}


log(f"=== gates {time.strftime('%Y-%m-%d %H:%M')} window {WIN}")
R = {}
R["R1"] = evaluate(101, n_facts=8, upd_frac=0.0, q_kinds=("cross",), kinds=["cross"])
R["R2"] = evaluate(102, n_facts=16, upd_frac=0.0, q_kinds=("cross",), kinds=["cross"])
R["R3"] = evaluate(103, n_facts=8, upd_frac=1.0, q_kinds=("cross",), kinds=["cross_upd"])
R["R4"] = evaluate(104, n_facts=8, upd_frac=0.0, q_kinds=("in",), kinds=["in"])
R["R5"] = evaluate(105, n_facts=32, upd_frac=0.0, q_kinds=("cross",), kinds=["cross"])
fmt = lambda r: "  ".join(f"{a} {100 * r[a]:.1f}%" for a in ARMS) + f"  (n={r['n_questions']})"
for k in R:
    log(f"{k}: {fmt(R[k])}")
R["pass"] = {
    "R1": R["R1"]["M"] >= 0.80 and R["R1"]["B0"] <= 0.10,
    "R2": R["R2"]["M"] >= R["R2"]["B1"] + 0.10,
    "R3": R["R3"]["M"] >= 0.70 and R["R3"]["M"] >= R["R3"]["B1"] + 0.20,
    "R4": R["R4"]["M"] >= R["R4"]["B0"] - 0.05,
}
for k, v in R["pass"].items():
    log(f"{k} -> {'PASS' if v else 'FAIL'}")
log("R1 by chunk gap, M: " + ", ".join(f"{g}: {100 * v:.0f}%" for g, v in R["R1"]["by_gap"]["M"].items()))
json.dump(R, open(os.path.join(RES, f"results{TAG}.json"), "w"), indent=1)
