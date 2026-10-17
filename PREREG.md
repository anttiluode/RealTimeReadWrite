# PREREG — RealTimeReadWrite

Committed before any code ran. Thresholds and predictions are not moved afterwards.

## The idea in four lines

A frozen language model gets one extra memory matrix S that is read and written on **every token**:

```
read      ŷ = S k                 what the memory expects at this address
residue   r = v − ŷ               what it got wrong
write     S ← α S + β r kᵀ        store only the surprise (α forget, β write strength, both from the token)
answer    o = S q                 query the just-updated memory, add o to the residual stream
```

No backward pass at use time: the write is one local gradient step on ‖S k − v‖². The language model's
own weights never change. Only the small adapter that makes k, v, q, α, β and maps o back is trained,
once, beforehand. After that, everything the memory learns, it learns live.

The question: can a frozen model with this memory recall facts it saw once, **beyond its own
context window**, and keep them current when they change?

## Part 1 — toy, trained and tested here (JAX, CPU)

**Language.** Tokens: 5 specials (PAD, FACT, IS, ASK, ANS), 128 entities, 32 values, 200 filler words.
A fact is `FACT e IS a`. A question is `ASK e ANS`, and the next token must be the current value of e.
Filler words are random and carry nothing.

**Base model (then frozen).** A 2-layer transformer, width 128, 4 heads, trained on 64-token chunks
where facts and questions about them appear inside the same chunk. It learns in-context lookup
inside its 64-token window. Then it is frozen for good.

**Beyond the window.** An episode is 8 chunks (512 tokens). Attention is blocked between chunks:
each chunk is processed as if the model had just been reset. Facts appear in early chunks; questions
about them come in later chunks, out of the model's sight. Values are random per episode, so no
weight can memorise them; only the live memory can carry them.

**Memory adapter.** Reads the residual stream after layer 1 (k, v, q, β, α are linear maps of it;
k and q are L2-normalised; β, α through a sigmoid), keeps S of size 64 x 64 carried across chunks,
and adds W_o · o to the residual stream before the final norm. About 50k trained numbers.

**Arms.** Same adapter shape, same training budget (3 000 steps, batch 16), same data:
- **B0** frozen base, no memory
- **B1** additive fast weights: S ← α S + β v kᵀ (linear-attention memory, no residue)
- **M** delta rule: S ← α S + β (v − S k) kᵀ  (main)

**Training data.** Episodes with 2-16 facts, 30% of them updated later with a new value, questions
both inside a chunk and across chunks. Loss on answer tokens only.

**Evaluation.** 1 000 fresh episodes per condition, never seen in training.

| gate | measure | pass if | my prediction |
|---|---|---|---|
| R1 | recall across chunks (beyond the window), 8 facts | M ≥ 80% and B0 ≤ 10% | M ~95%, B0 ~3% (1/32 chance) |
| R2 | recall across chunks, 16 facts (interference) | M ≥ B1 + 10 points | M ~90%, B1 ~70% |
| R3 | after an update, recall of the **latest** value | M ≥ 70% and M ≥ B1 + 20 points | M ~85%, B1 ~45% |
| R4 | recall inside a chunk (what the frozen base already does) | M ≥ B0 − 5 points | equal |
| R5 | 32 facts (twice the training maximum), reported only | — | M degrades gracefully |

**Kill condition.** R1 fails on M: a frozen model with this memory cannot recall beyond its window
even in the easiest setting; the four lines do not deliver live read-write here.
If R3 fails while R1 passes, the delta rule's distinctive part (writing only the residue) adds
nothing over plain fast weights in this setting.

## Part 2 — a real frozen LLM, for a GPU (not run here)

`qwen/` applies the same adapter to Qwen2.5-0.5B, frozen: memory reads the residual stream at a middle
layer and writes before the final norm; text is fed in chunks with the KV cache cleared between chunks,
so anything recalled across chunks came through the memory. Facts are sentences like "The favourite
animal of Mira Halden is the tiger." with single-token answers, buried in generated filler text.

The build machine has no GPU and cannot download model weights, so Part 2 is shipped as a protocol
plus code smoke-tested on a randomly initialised miniature Qwen2. Its gates are the same R1-R4,
measured on the real model when run.
