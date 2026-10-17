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

PREREG_2 gates: see `results/gate_log_w3.txt` (filled in below when the run finished).

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

## Untested next steps

- **Keys from what the model already computes.** Use the frozen model's own attention keys/queries at
  the read layer (already trained to address "what is this about") as k and q, instead of learning new
  projections.
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
