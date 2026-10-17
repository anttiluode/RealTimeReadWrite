# PREREG_2 — the memory reads a short local window

Written while the PREREG.md run was still training, and committed before any code for this change.

## Why

A diagnostic during the registered run (`results/diagnostic_log.txt`, linear probes on 24 000
examples each) found that the frozen base's layer-1 state does not carry the entity where the
memory reads and writes:

| position | entity decodable (chance 0.8%) |
|---|---|
| the entity token itself | 90.2% |
| the fact's value token (memory writes) | 1.9% |
| the question's ANS token (memory reads) | 2.0% |

So in PREREG.md's design the memory has no key for "which entity". That run is completed and
reported as registered regardless of this.

## The one change

k, v, q, β and α are computed from the concatenation [h1(t), h1(t−1), h1(t−2)] instead of h1(t)
alone: a causal window of three positions (zeros before the episode start; the window may cross a
chunk boundary, since the adapter is not the base model). Everything else is identical to
PREREG.md: base model (the same frozen file), S size, rules, training budget, data, seeds,
evaluation, gates R1-R5 and their thresholds.

## Predictions

R1 passes for the delta rule (≥ 80%). R3 is the real test of the residue: the delta rule beats
additive on updated facts by ≥ 20 points. R4: equal. If R1 still fails, a key alone is not enough
and the four lines do not deliver live recall in this setting.
