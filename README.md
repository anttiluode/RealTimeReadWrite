# RealTimeReadWrite

**Constant read-write for a frozen language model.** One memory matrix S, read and written on every
token, with no backward pass at use time:

```
read      ŷ = S k                 what the memory expects at this address
residue   r = v − ŷ               what it got wrong
write     S ← α S + β r kᵀ        store only the surprise (α forget, β write strength, both from the token)
answer    o = S q                 query the just-updated memory; add o to the residual stream
```

The write is one local gradient step on ‖S k − v‖²: the delta rule (Widrow & Hoff, 1960). The same
four lines run inside today's strongest recurrent sequence models (Gated DeltaNet; Titans, test-time
training layers and fast-weight linear attention are relatives). This repo asks whether they can be
bolted onto a **frozen** model, so that it recalls facts it saw once beyond its own context window
and keeps them current when they change.

## What is here

| part | what | status |
|---|---|---|
| `toy/` | a small frozen transformer (64-token window) + the memory, trained and tested on CPU | run, results below |
| `qwen/jspace.py` | Gate 4: native J-space memory on frozen Qwen2.5-0.5B-Instruct, no trained parameters (PREREG_3) | run: G4.2 PASS, G4.1 and G4.4 FAIL, G4.3 uninformative (below) |
| `qwen/rtrw.py` | the same memory on frozen **Qwen2.5-0.5B**: reads a middle layer, writes before the final norm, text fed in chunks with the KV cache cleared between them | written for a GPU; full train → gates path smoke-tested on a miniature random Qwen2, **not run on real weights** |

```
# toy (CPU, ~1.5 h)
python toy/train.py && python toy/gates.py
python toy/train.py --window 3 && python toy/gates.py --window 3

# real model (GPU)
pip install torch transformers
python qwen/rtrw.py train --rule delta    && python qwen/rtrw.py train --rule additive
python qwen/rtrw.py gates
python tests/test_qwen_smoke.py            # CPU, no download
```

## Results on the toy (pre-registered in PREREG.md, before any code)

**Every gate failed.** The live memory did not give the frozen model recall beyond its window.

| gate | no memory (B0) | additive (B1) | delta rule (M) | |
|---|---|---|---|---|
| R1 recall across chunks, 8 facts | 3.0% | 19.7% | 14.9% | FAIL (needs ≥ 80%) |
| R2 16 facts | 3.1% | 11.5% | 7.4% | FAIL |
| R3 latest value after an update | 3.3% | 16.0% | 11.8% | FAIL |
| R4 recall inside the window | 85.3% | 66.0% | 72.9% | FAIL: the memory **costs** 12 points |
| R5 32 facts (reported) | 3.3% | 7.1% | 4.6% | |

Chance is 3.1% (32 values). Recall by distance for M: 29% one chunk later, 2% seven chunks later.

### Why: diagnostics, in the order they were run (`results/diagnostic_log.txt`)

1. **The frozen base had no key where the memory looked.** Linear probes on 24 000 examples: the
   entity a fact is about is 90% decodable from layer 1 at the entity's own token, but 1.9% at the
   fact's value token (where the memory writes) and 2.0% at the question's answer position (where it
   reads). The memory had nothing to address by. This also explains the base's own weak recall:
   it never really looked up by entity.
2. **Forgetting was not the problem.** The trained gates are selective: forget gate 0.9996 on filler
   (a fact keeps 83% of its strength over 448 tokens), write gate 0.97 on fact values versus 0.16 on
   filler.
3. **With the key in reach, the memory stored the right content at the wrong address.** PREREG_2
   (committed before coding it) let the memory read a 3-token window, so the entity is two positions
   back at a fact and one back at a question. The trained window memory writes values from which the
   right answer is 96% decodable, but a question's query matches the right fact's key no better than
   any other fact's (cosine 0.321 vs 0.318; best match correct in 14.6% against 12.5% chance).
   Learning the address through 512 steps of reads and writes settled on a lazy optimum: blend
   recent values.

PREREG_2 gates (window of 3), all fail as well:

| gate | B0 | B1 additive | M delta | |
|---|---|---|---|---|
| R1 across chunks, 8 facts | 3.0% | 20.5% | 22.1% | FAIL |
| R2 16 facts | 3.1% | 12.5% | 11.2% | FAIL |
| R3 latest value after an update | 3.3% | 17.1% | 17.4% | FAIL |
| R4 inside the window | 85.3% | 74.0% | 78.9% | FAIL (cost now 6 points) |
| R5 32 facts | 3.3% | 8.1% | 7.2% | |

The window helped a little (R1 14.9 → 22.1%, smaller in-window cost), and the delta rule's
distinctive advantage on updated facts (R3) never appeared, because nothing was stored at the right
address to update.

### Reading it

- The four lines are not what failed. Given a usable key, the delta rule's algebra does what it
  says (`tests/test_qwen_smoke.py` checks it directly: repeating a known fact writes nothing).
- **The hard part is the address.** A memory bolted onto a frozen model needs the model to already
  carry "what is this about" at the positions the memory reads, *and* the adapter has to learn to
  read it, through a long recurrence where the gradient for keys is weak. Models trained with the
  memory inside from the start (DeltaNet, Titans) learn their keys and their representations
  together. This frozen-retrofit setup only learns the keys.
- **A cost to watch:** a memory without a good key injects noise and makes the frozen model worse at
  what it already did (R4, −12 points).
- **What a real LLM changes:** large models do carry subject identity forward in the residual
  stream. That is how they answer questions about earlier text. So point 1 may not bite on Qwen;
  point 3, learning the keys, still might. That is what `qwen/rtrw.py gates` measures.

## Gate 4: a native memory in the model's own workspace (PREREG_3, run 2026-10-18)

Both the toy above and Sol's [LiveMode](https://github.com/anttiluode/LiveMode) point at the same
gap. LiveMode's readout works on real Qwen2.5-0.5B-Instruct (100% on 400 held-out trials), but its
address is a SHA-256 hash of the name string, the colour codes are supplied, and a 3,584-parameter
adapter is trained. This repo's memory had to learn its address and never did.

Gate 4 removes all three crutches. It uses the **J-space** of Gurnee, Sofroniew, Lindsey et al.
(2026, arXiv 2607.15495, "Verbalizable Representations Form a Global Workspace in Language
Models"): the directions a model is poised to verbalize, which carry concepts in a middle band of
layers and act as arguments for downstream computation.

```
J_l = E_{t, t'>=t, prompt} [ d h_final,t' / d h_l,t ]       the paper's J-lens, one d x d matrix per layer
v_x = J_l^T (gamma * u_x)                                    J-lens vector of token x
key   = J-space component of the state just before a value   (k-sparse, nonnegative, whole vocabulary)
value = J-space component of the state at the value
write S <- S + (v - S k) k^T       read o = S k_q       inject h <- h + alpha |h| o/|o|
```

No parameter is trained. From dev names only, the gate selects a read layer, a write layer, one
gain, and two mean vectors used to centre the keys.

| gate | question | pass |
|---|---|---|
| G4.1 | Does the question's last token carry *who* in the J-space? (zero-parameter probe, 48 names) | ≥ 90% |
| G4.2 | Does adding a colour's J-lens vector flip the answer? (no fact anywhere; random direction as control) | ≥ 90%, random ≤ 40% |
| G4.3 | Does the model *reason* with the injected colour? ("Is X the same color as a ripe banana?") | ≥ 75%, if the in-context ceiling ≥ 75% |
| G4.4 | Live memory: facts, overwrites and filler, then late questions, 8 / 16 / 32 names | ≥ 80% at 8, ≥ 60% at 32 |

G4.4 also reports these arms:
- stale memory;
- no memory;
- facts in context;
- uncentred keys;
- an oracle key (the name's own J-lens vector, the analogue of LiveMode's hash);
- a wrong-address read.

```
python qwen/jspace.py lens     # the J-lens for all 24 layers: 128 prompts x 896 rows, ~30-60 min on an RTX 3060
python qwen/jspace.py gates    # G4.1-G4.4 (~30 min); log in results/gate4_log.txt
python tests/test_jspace_smoke.py   # CPU, no download
```

The smoke test checks the J-lens against a brute-force full Jacobian on a miniature Qwen2 (max
error 9e-8 at every layer). It also checks that pursuit recovers a planted sparse code exactly, the
delta-rule overwrite, and the full lens → gates path.

### Results on Qwen2.5-0.5B-Instruct (RTX 3060, run by Antti; 128-prompt J-lens, about 32 min)

**One gate passes, two fail, one is uninformative.** The memory itself failed.

| gate | registered result | verdict |
|---|---|---|
| G4.1 address at the question token | read layer 21 (selected on dev, tie with 23 broken to the earlier): test **82.6%** (chance 2.1%) | **FAIL** (needs 90%) |
| G4.2 write-back, layer 16, α = 0.5 | J **100%**, logit lens 100%, random direction 30.1%, none 0% (by construction), [name-token site 56.7%, band 87.0%, reported only] | **PASS** |
| G4.3 reason with the injected colour | ceiling (fact in context) **50.0%**, J 41.7%, logit 66.7%, none 52.1%, [band 50.0%] | **UNINFORMATIVE**: the 0.5B model cannot do it even when told |
| G4.4 live memory | see below | **FAIL** at N = 8 |

| G4.4, latest colour after one overwrite each (chance 25%) | L (live) | stale | none | ctx (all in prompt) | L_raw | oracle key | wrong address |
|---|---|---|---|---|---|---|---|
| N = 8 | 23.1% | 21.2% | 20.0% | 30.6% | 25.6% | 35.6% | 21.2% |
| N = 16 | 23.1% | 22.5% | 25.0% | 29.4% | 25.9% | 34.1% | 25.3% |
| N = 32 | not completed (stopped; the gate had already failed at N = 8) | | | | | | |

#### What the run shows

1. **The address is not in the workspace where the question is asked.**
   - At the name token, the J-lens names the person 100% of the time at layers 0–15. That is the
     token's own identity.
   - At the question's last token it is at chance through layer 15, and 12–18% at layers 16–19
     (the middle band, the workspace).
   - "Who" appears at the question token only in the last four layers (60–92%). There the plain
     logit lens does almost as well, so it looks like output preparation, not an address. Layer 23
     reaches 91.7% on test, but it was not the registered choice.
2. **Writing is easy.** At layers 15–19, adding a colour's J-lens vector at half the residual norm
   flips the answer every time. The plain unembedding row (logit lens) does the same at layer 16;
   on this model the J-lens adds nothing for steering.
3. **Writing at the name token partly reaches the answer (56.7%, reported only).** A colour planted
   on the subject's own position at layer 16 is carried to the answer by the model's own attention
   in over half the trials.
4. **The live memory does nothing useful, and the oracle key barely helps (35.6%).** The address is
   not the only failure. The stored *value* is the J-space content at the colour token, which is
   what the model computes after reading "yellow" (it is predicting the full stop). That is not the
   direction that makes it *say* yellow. The task itself is also hard for this model: with every
   fact, overwrite and filler sentence in the prompt it scores only 30.6%. Sol's LiveMode context
   arm, with one fact, scores 100% four-colour.

So "experience changes what the model says" works natively, but "experience is stored and found
again" does not compose from the J-space at 0.5B. Notes, from Sol and from this run's own numbers:
- The memory here is d × d (3.2 MB), against LiveMode's 1 KB. They are not equal-budget memories.
- The G4.3 logit-over-J difference (66.7% vs 41.7%, 24 items per class) is suggestive at best.

## Untested next steps

- **Keys from what the model already computes.** Use the frozen model's own attention keys/queries at
  the read layer (already trained to address "what is this about") as k and q, instead of learning new
  projections. Gate 4 takes the J-space route to the same idea.
- **A curriculum.** Train on one fact and one question one chunk apart before longer episodes, so
  the key gradient is strong before interference sets in.
- **Gated filler.** A write gate that is closed by default (bias −4), so filler does not erode stored facts.

Each would be its own pre-registered run.

## Corrections during the run

- The first training launch failed because `results/` did not exist; relaunched, nothing ran.
- `toy/gates.py` crashed writing JSON (numpy booleans) after printing every number; fixed and rerun
  (deterministic, identical numbers; both in `results/gate_log.txt`).

## Prior art

Widrow & Hoff 1960 (delta rule); Schmidhuber 1992 and Schlag, Irie & Schmidhuber 2021 (fast weights,
*Linear Transformers Are Secretly Fast Weight Programmers*); Yang, Kautz & Hatamizadeh 2025
(*Gated Delta Networks*); Sun et al. 2024 (test-time training layers); Behrouz, Zhong & Mirrokni 2025
(*Titans*); memory-augmented and retrieval models (Wu et al. 2022, *Memorizing Transformers*).
