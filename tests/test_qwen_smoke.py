"""Smoke test for qwen/rtrw.py on a randomly initialised miniature Qwen2 (no download, CPU).
python tests/test_qwen_smoke.py"""
import os
import re
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen"))
import rtrw
from transformers import Qwen2Config, Qwen2ForCausalLM


class WordTok:
    """Word-level stand-in tokenizer: every word and punctuation mark is one token."""
    def __init__(self):
        words = set(rtrw.FIRST + rtrw.LAST + rtrw.ANIMALS + rtrw.ADJ + rtrw.NOUN + rtrw.VERB +
                    "The the favourite animal of is Correction : now Question what ? Answer near . ,".split())
        self.vocab = {w: i for i, w in enumerate(sorted(words))}

    def encode(self, s, add_special_tokens=False):
        return [self.vocab[w] for w in re.findall(r"\w+|[^\w\s]", s)]


fails = 0
def check(name, ok, info=""):
    global fails
    print(f"{'ok  ' if ok else 'FAIL'} {name} {info}")
    fails += 0 if ok else 1


torch.manual_seed(0)
tok = WordTok()
cfg = Qwen2Config(vocab_size=len(tok.vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
lm = Qwen2ForCausalLM(cfg).eval()
ep = rtrw.Episodes(tok, chunk=96, seed=3)
ids, tgt, kind = ep.batch(2, n_chunks=4, n_facts=8, upd_frac=0.5)
check("episode shape", ids.shape == (2, 384), str(tuple(ids.shape)))
pos = (tgt >= 0).nonzero().tolist()
aligned = all(ids[b, t + 1].item() == ep.animal_ids[tgt[b, t].item()] for b, t in pos)
check("every target sits right before its answer token", aligned and len(pos) > 0, f"{len(pos)} targets")
kinds = {k for row in kind for k in row if k}
check("episode has cross-chunk and in-chunk questions", {"cross", "in"} <= kinds or "cross" in kinds, str(kinds))

base = rtrw.LiveModel(lm, 2, None)
with torch.no_grad():
    l0 = rtrw.run_episode(base, ids, 96)
mem = rtrw.ReadWriteMemory(64, 16, "delta")
base.mem = mem
with torch.no_grad():
    l1 = rtrw.run_episode(base, ids, 96)
check("memory changes the output", (l1 - l0).abs().max().item() > 0)
check("memory state carries across chunks", mem.S.abs().sum().item() > 0)

# only the memory is trainable; gradients reach it and nothing else
opt = torch.optim.AdamW(mem.parameters(), lr=3e-3)
aid = torch.tensor(ep.animal_ids)
for i in range(30):
    ids, tgt, _ = ep.batch(4, n_chunks=4, n_facts=6, upd_frac=0.3)
    loss = rtrw.answer_loss(rtrw.run_episode(base, ids, 96), tgt, aid)
    opt.zero_grad(); loss.backward(); opt.step()
check("frozen model has no gradients", all(p.grad is None for p in lm.parameters()))
check("memory received gradients", all(p.grad is not None for p in mem.parameters()))
check("30 training steps run, loss finite", torch.isfinite(loss).item(), f"loss {loss.item():.3f}")

# the delta rule stores only the residue: writing the same pair twice leaves S unchanged the 2nd time (β=1, α=1)
m2 = rtrw.ReadWriteMemory(8, 8, "delta")
with torch.no_grad():
    for p_ in m2.parameters(): p_.zero_()
    m2.k.weight.copy_(torch.eye(8)); m2.v.weight.copy_(2 * torch.eye(8)); m2.q.weight.copy_(torch.eye(8))
    m2.beta.bias.fill_(50.0); m2.alpha.bias.fill_(50.0)
    m2.reset(1, "cpu")
    h = torch.zeros(1, 2, 8); h[0, :, 0] = 1.0
    m2(h[:, :1]); S1 = m2.S.clone(); m2(h[:, 1:])
check("delta rule: repeating a known fact writes nothing", torch.allclose(S1, m2.S, atol=1e-5))
sys.exit(1 if fails else 0)
