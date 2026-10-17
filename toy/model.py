"""Frozen base transformer + the live read-write memory (PREREG.md Part 1). Pure JAX.

Chunks: positions restart every 64 tokens and attention never crosses a chunk boundary,
so each chunk is seen as if the model had just been reset. Only the memory crosses.
"""
import math

import jax
import jax.numpy as jnp

from task import CHUNK, VOCAB, V0, NV

D, H, L = 128, 4, 2
DK = 64


def init_base(key):
    ks = iter(jax.random.split(key, 32))
    n = lambda shape, s=0.02: s * jax.random.normal(next(ks), shape)
    p = {"tok": n((VOCAB, D)), "pos": n((CHUNK, D)), "lnf_g": jnp.ones(D), "lnf_b": jnp.zeros(D), "blocks": []}
    for _ in range(L):
        p["blocks"].append({"ln1_g": jnp.ones(D), "ln1_b": jnp.zeros(D), "qkv": n((D, 3 * D)), "o": n((D, D)),
                            "ln2_g": jnp.ones(D), "ln2_b": jnp.zeros(D), "w1": n((D, 4 * D)), "b1": jnp.zeros(4 * D),
                            "w2": n((4 * D, D)), "b2": jnp.zeros(D)})
    return p


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5) * g + b


def _block(blk, x, mask):
    B, T, _ = x.shape
    h = _ln(x, blk["ln1_g"], blk["ln1_b"])
    q, k, v = jnp.split(h @ blk["qkv"], 3, -1)
    sh = lambda z: z.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)
    q, k, v = sh(q), sh(k), sh(v)
    a = jnp.where(mask, q @ k.transpose(0, 1, 3, 2) / math.sqrt(D // H), -1e9)
    y = (jax.nn.softmax(a, -1) @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    x = x + y @ blk["o"]
    h = _ln(x, blk["ln2_g"], blk["ln2_b"])
    return x + jax.nn.gelu(h @ blk["w1"] + blk["b1"]) @ blk["w2"] + blk["b2"]


def base_hidden(p, toks):
    """Returns (h1, h2): residual stream after block 1 and after block 2."""
    B, T = toks.shape
    nc = T // CHUNK                       # chunks never see each other: run them as separate sequences
    c = toks.reshape(B * nc, CHUNK)
    t = jnp.arange(CHUNK)
    mask = t[:, None] >= t[None, :]
    x = p["tok"][c] + p["pos"][t]
    h1 = _block(p["blocks"][0], x, mask)
    h2 = _block(p["blocks"][1], h1, mask)
    return h1.reshape(B, T, D), h2.reshape(B, T, D)


def value_logits(p, h):
    """Logits over the 32 value tokens only (all gates score the restricted answer)."""
    z = _ln(h, p["lnf_g"], p["lnf_b"])
    return z @ p["tok"][V0:V0 + NV].T


# ------------------------------------------------------------------ the memory
def init_mem(key, win=1):
    """win = how many positions the memory reads: [h1(t), h1(t-1), ...] (PREREG_2 uses 3)."""
    ks = iter(jax.random.split(key, 8))
    n = lambda shape, s: s * jax.random.normal(next(ks), shape)
    Din = D * win
    return {"Wk": n((Din, DK), 1 / math.sqrt(Din)), "Wv": n((Din, DK), 1 / math.sqrt(Din)), "Wq": n((Din, DK), 1 / math.sqrt(Din)),
            "wb": jnp.zeros(Din), "bb": jnp.zeros(()), "wa": jnp.zeros(Din), "ba": jnp.full((), 4.0),
            "Wo": n((DK, D), 0.02)}


def window(h1, win):
    """Causal window [h(t), h(t-1), ..., h(t-win+1)], zeros before the episode start."""
    if win == 1:
        return h1
    parts = [h1] + [jnp.pad(h1, ((0, 0), (j, 0), (0, 0)))[:, :-j] for j in range(1, win)]
    return jnp.concatenate(parts, -1)


def memory(m, h1, rule):
    """Read and write on every token. rule: 'delta' (write the residue) or 'additive' (write the value).
    Returns o (B,T,DK) and the final S."""
    nrm = lambda z: z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)
    h1 = window(h1, m["Wk"].shape[0] // D)          # window size is read off the adapter's shape
    k, q = nrm(h1 @ m["Wk"]), nrm(h1 @ m["Wq"])
    v = h1 @ m["Wv"]
    beta = jax.nn.sigmoid(h1 @ m["wb"] + m["bb"])
    alpha = jax.nn.sigmoid(h1 @ m["wa"] + m["ba"])

    def step(S, inp):
        kt, vt, qt, bt, at = inp
        pred = jnp.einsum("bij,bj->bi", S, kt)                 # read:    ŷ = S k
        w = (vt - pred) if rule == "delta" else vt              # residue: r = v − ŷ   (additive: v)
        S = at[:, None, None] * S + bt[:, None, None] * jnp.einsum("bi,bj->bij", w, kt)   # write
        return S, jnp.einsum("bij,bj->bi", S, qt)              # answer:  o = S q

    B = h1.shape[0]
    S0 = jnp.zeros((B, DK, DK))
    xs = (k.transpose(1, 0, 2), v.transpose(1, 0, 2), q.transpose(1, 0, 2), beta.T, alpha.T)
    S, o = jax.lax.scan(step, S0, xs)
    return o.transpose(1, 0, 2), S


def forward(p, m, toks, rule=None):
    h1, h2 = base_hidden(p, toks)
    if rule is None:
        return value_logits(p, h2)
    o, _ = memory(m, h1, rule)
    return value_logits(p, h2 + o @ m["Wo"])


def answer_loss(logits, tgt):
    msk = tgt >= 0
    lp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(lp, jnp.maximum(tgt, 0)[..., None], -1)[..., 0]
    return (nll * msk).sum() / jnp.maximum(msk.sum(), 1)


def adam(params, grads, st, lr, b1=0.9, b2=0.99):
    t = st["t"] + 1
    mm = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, st["m"], grads)
    vv = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * g * g, st["v"], grads)
    params = jax.tree_util.tree_map(lambda p_, a, b: p_ - lr * (a / (1 - b1 ** t)) / (jnp.sqrt(b / (1 - b2 ** t)) + 1e-8), params, mm, vv)
    return params, {"m": mm, "v": vv, "t": t}


def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, p), "t": 0}
