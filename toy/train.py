"""Train the base (then freeze it) and the two memory adapters. python toy/train.py
Writes toy/cache/*.pkl and results/train_log.txt."""
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
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RES, exist_ok=True)
LOG = open(os.path.join(RES, "train_log.txt"), "a")


def log(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def acc(logits, tgt):
    msk = tgt >= 0
    return float(((np.argmax(logits, -1) == tgt) & msk).sum() / max(msk.sum(), 1))


def train_base(steps=2500, B=64):
    rng = np.random.default_rng(0)
    p = M.init_base(jax.random.PRNGKey(0))
    st = M.adam_init(p)

    @jax.jit
    def step(p, st, toks, tgt, lr):
        l, g = jax.value_and_grad(lambda p: M.answer_loss(M.forward(p, None, toks), tgt))(p)
        p, st = M.adam(p, g, st, lr)
        return p, st, l

    t0 = time.time()
    for i in range(steps):
        toks, tgt, _, _ = batch(rng, B, n_chunks=1, n_facts=int(rng.integers(2, 7)), q_kinds=("in",))
        lr = 2e-3 * min(1, (i + 1) / 200) * 0.5 * (1 + np.cos(np.pi * i / steps))
        p, st, l = step(p, st, jnp.asarray(toks), jnp.asarray(tgt), lr)
        if (i + 1) % 250 == 0:
            vt, vg, _, _ = batch(np.random.default_rng(99), 256, n_chunks=1, n_facts=4, q_kinds=("in",))
            log(f"base step {i + 1}: loss {float(l):.3f}  in-window recall {acc(np.asarray(M.forward(p, None, jnp.asarray(vt))), vg):.3f}  ({time.time() - t0:.0f}s)")
    pickle.dump(jax.tree_util.tree_map(np.asarray, p), open(os.path.join(CACHE, "base.pkl"), "wb"))
    return p


def train_mem(p, rule, steps=3000, B=16, win=1):
    rng = np.random.default_rng(1)
    m = M.init_mem(jax.random.PRNGKey(1), win)
    st = M.adam_init(m)
    p = jax.tree_util.tree_map(jnp.asarray, p)                 # frozen: never passed to the optimiser

    @jax.jit
    def step(m, st, toks, tgt, lr):
        l, g = jax.value_and_grad(lambda m: M.answer_loss(M.forward(p, m, toks, rule), tgt))(m)
        m, st = M.adam(m, g, st, lr)
        return m, st, l

    t0 = time.time()
    for i in range(steps):
        toks, tgt, _, _ = batch(rng, B, n_chunks=8, n_facts=int(rng.integers(2, 17)), upd_frac=0.3)
        lr = 3e-3 * min(1, (i + 1) / 200) * 0.5 * (1 + np.cos(np.pi * i / steps))
        m, st, l = step(m, st, jnp.asarray(toks), jnp.asarray(tgt), lr)
        if (i + 1) % 500 == 0:
            vt, vg, vk, _ = batch(np.random.default_rng(98), 64, n_chunks=8, n_facts=8, q_kinds=("cross",))
            log(f"memory[{rule}{TAG}] step {i + 1}: loss {float(l):.3f}  cross-chunk recall {acc(np.asarray(M.forward(p, m, jnp.asarray(vt), rule)), vg):.3f}  ({time.time() - t0:.0f}s)")
    pickle.dump(jax.tree_util.tree_map(np.asarray, m), open(os.path.join(CACHE, f"mem_{rule}{TAG}.pkl"), "wb"))


WIN = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 1
TAG = "" if WIN == 1 else f"_w{WIN}"

if __name__ == "__main__":
    log(f"window {WIN}")
    log(f"=== train {time.strftime('%Y-%m-%d %H:%M')}")
    bp = os.path.join(CACHE, "base.pkl")
    base = pickle.load(open(bp, "rb")) if os.path.exists(bp) else train_base()
    for rule in ("delta", "additive"):
        if not os.path.exists(os.path.join(CACHE, f"mem_{rule}{TAG}.pkl")):
            train_mem(base, rule, win=WIN)
