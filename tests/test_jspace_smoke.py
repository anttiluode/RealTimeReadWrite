"""Smoke test for qwen/jspace.py (Gate 4) on a randomly initialised miniature Qwen2 (no download, CPU).
Checks the J-lens against a brute-force Jacobian, the pursuit on a planted sparse code, the delta-rule
memory, and the full lens -> gates path. Numbers from a random model mean nothing.
python tests/test_jspace_smoke.py"""
import os
import re
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen"))
import jspace as js
from transformers import Qwen2Config, Qwen2ForCausalLM


class WordTok:
    """Word-level stand-in tokenizer: every word, number and punctuation mark is one token."""
    def __init__(self):
        text = " ".join(js.NAME_POOL + js.COLORS + list(js.OBJECTS.values()) + js.QUESTIONS + [js.FACT, js.YESNO]
                        + js.FILLER + js.OPENINGS + ["Yes", "No", "n", "c", "o"])
        words = set(re.findall(r"\w+|[^\w\s]", text))
        self.vocab = {w: i for i, w in enumerate(sorted(words))}

    def encode(self, s, add_special_tokens=False):
        return [self.vocab[w] for w in re.findall(r"\w+|[^\w\s]", s)]


fails = 0
def check(name, ok, info=""):
    global fails
    print(f"{'ok  ' if ok else 'FAIL'} {name} {info}", flush=True)
    fails += 0 if ok else 1


torch.manual_seed(0)
tok = WordTok()
cfg = Qwen2Config(vocab_size=len(tok.vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=True)
lm = Qwen2ForCausalLM(cfg).eval()
fz = js.Frozen(lm, tok, "cpu")

# 1. J-lens against the brute-force Jacobian of h_final with respect to h_l, averaged over t' >= t
prompts = js.corpus(fz, 2, 7, seed=0)
check("corpus shape", tuple(prompts.shape) == (2, 7), str(tuple(prompts.shape)))
Js = js.jlens(fz, prompts, rows_per_pass=8, log=lambda s: None)
for l in range(fz.L):
    acc, pairs = torch.zeros(32, 32), 0
    for p in prompts:
        _, cap = fz.run(p, capture=[l])
        T = p.shape[0]

        def f(x, l=l, p=p):
            _, c = fz.run(p, capture=["final"], delta={l: x[None]})
            return c["final"][0]

        Jfull = torch.autograd.functional.jacobian(f, torch.zeros(T, 32))      # (T', d, T, d)
        for t in range(T):
            for t2 in range(t, T):
                acc += Jfull[t2, :, t, :]
        pairs += T * (T + 1) // 2
    err = (Js[l] - acc / pairs).abs().max().item()
    check(f"J-lens layer {l} equals brute-force Jacobian average", err < 1e-5, f"max err {err:.2e}")
check("last layer's J is a multiple of the identity", torch.allclose(Js[fz.L - 1], Js[fz.L - 1][0, 0] * torch.eye(32), atol=1e-6))

# 2. J-lens vectors are rows of W_U J (with the final-norm gain) and unit length
ids = [js.tid(fz, c) for c in js.COLORS]
V = js.lens_vectors(fz, Js[1], ids)
ref = torch.nn.functional.normalize((fz.gamma[None] * fz.U[ids]) @ Js[1], dim=-1)
check("lens_vectors = normalize(gamma*u @ J)", torch.allclose(V, ref, atol=1e-6))

# 3. pursuit recovers a planted nonnegative sparse code from a nearly orthogonal dictionary
g = torch.Generator().manual_seed(1)
D = torch.nn.functional.normalize(torch.randn(400, 256, generator=g), dim=-1)
jsp = types.SimpleNamespace(D=D, k=8)
true = [3, 77, 150, 299]
h = D[true].T @ torch.tensor([2.0, 1.5, 1.0, 0.7])
comp, sel, coef = js.JSpace.component(jsp, h, iters=200)
check("pursuit finds the planted atoms", set(true) <= set(sel), f"selected {sorted(sel)[:6]}")
check("pursuit reconstructs the activation", (comp - h).norm().item() / h.norm().item() < 0.05,
      f"rel err {(comp - h).norm().item() / h.norm().item():.3f}")
check("pursuit coefficients are nonnegative", bool((coef >= 0).all()))

# 4. delta rule: overwrite replaces the stored value; repeating a known pair writes nothing
mem = js.NativeMemory(16, "cpu")
k = torch.nn.functional.normalize(torch.randn(16, generator=g), dim=0)
v1, v2 = torch.randn(16, generator=g), torch.randn(16, generator=g)
mem.write(k, v1); S1 = mem.S.clone(); mem.write(k, v1)
check("repeating a known pair writes nothing", torch.allclose(S1, mem.S, atol=1e-6))
mem.write(k, v2)
check("overwrite returns the new value", torch.allclose(mem.read(k), v2, atol=1e-5))

# 5. injection reaches the output only through the chosen layer and position
p = torch.tensor(fz.enc(js.QUESTIONS[0].format(n="James")))
out0, cap = fz.run(p, capture=[1])
d = torch.zeros_like(cap[1][0]); d[-1] = 5.0
out1, _ = fz.run(p, delta={1: d[None]})
check("injection changes the last position only", (out1[0, :-1] - out0[0, :-1]).abs().max() < 1e-5
      and (out1[0, -1] - out0[0, -1]).abs().max() > 1e-3)

# 6. names: single-token filter and split
names = js.single_token_names(fz)
dev, test = js.split(names)
check("64 names, 16 dev / 48 test, disjoint", len(names) == 64 and len(dev) == 16 and len(test) == 48
      and not set(dev) & set(test))

# 7. the full lens -> gates command path on the miniature model
js.load = lambda args: (tok, lm)
args = types.SimpleNamespace(model="mini-smoke", device="cpu", n_prompts=3, T=8, rows=16, k=4, sizes=[4, 8], episodes=1)
res_dir_log = os.path.join(js.RES, "gate4_log.txt")
had_log = os.path.exists(res_dir_log)
R = js.cmd_gates(args, fz)
check("gates run end to end", set(R["pass"]) == {"G4.1", "G4.2", "G4.3", "G4.4"}, str(R["pass"]))
check("G4.2 none arm is 0% by construction (targets exclude the model's own answer)", R["G4.2"]["test"]["none"]["four"] == 0.0)
check("G4.4 reports every arm", {"L", "stale", "none", "ctx", "L_raw", "oracle_key", "wrong_address_hit"} <= set(R["G4.4"][4]))
os.remove(js.cache_path(args))
os.remove(os.path.join(js.HERE, "results_jspace.json"))
if not had_log:
    os.remove(res_dir_log)
sys.exit(1 if fails else 0)
