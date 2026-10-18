# PREREG_3 — Gate 4: a native read-write memory in the model's own workspace

Written and committed before any code for it. Nothing here has been run on real weights when this
file is committed.

## Why

Two earlier results point the same way.

- This repo (PREREG, PREREG_2): a learned memory bolted onto a frozen model stored the right
  values but never learned the address (query-to-fact key cosine 0.321 vs 0.318 for other facts).
- anttiluode/LiveMode (Sol): the readout half works on real Qwen2.5-0.5B-Instruct (Gate 2, 100% on
  400 held-out trials), but the address is a SHA-256 hash of the name string, the four colour
  codes are supplied, and a 3,584-parameter adapter is trained. Gate 3 falls to 38% at 32 slots.

Gurnee, Sofroniew, Lindsey et al. (arXiv 2607.15495, "Verbalizable Representations Form a Global
Workspace in Language Models") identify the J-space: the directions a model is poised to verbalize,
which carry concepts in a middle band of layers and are passed as arguments to downstream
computation. If that holds for Qwen, the model already provides the address (who is this about)
and the write format (a concept the rest of the network knows how to read). Gate 4 tests whether a
memory with **no hash, no supplied codebook and no trained parameters** can live there.

## Model and lens

- Model: `Qwen/Qwen2.5-0.5B-Instruct` (Sol's model, for direct comparison), fp32, frozen.
- h_ℓ = output of decoder layer ℓ (ℓ = 0..23); h_final = the input to the final norm.
- J-lens exactly as the paper's §2.1: J_ℓ = E_{t, t'≥t, prompt}[∂h_final,t' / ∂h_ℓ,t], one d×d matrix
  per layer. Corpus: 128 prompts of 64 tokens sampled from the model itself (temperature 1, seed 0)
  after 32 neutral openings. The paper used 1,000 pretraining-like prompts and reports robustness to
  the count (§A.7); this is the deviation.
- J-lens vector of token x at layer ℓ: v_x = J_ℓᵀ (γ ⊙ u_x), with u_x the unembedding row and γ the
  final RMSNorm weight. v̂ = v / ‖v‖.
- J-space component of an activation (paper §2.3): nonnegative k-sparse combination of unit J-lens
  vectors over the **whole vocabulary**, k = 16, chosen by matching pursuit with a nonnegative
  projected-gradient refit on the support. (The paper uses gradient pursuit; this is a close
  variant.)

## Names, colours, prompts

- Names: a fixed pool of common first names, in the order listed in the code; keep those whose
  " Name" is one token; take the first 64; dev = positions 0, 4, 8, … (16 names), test = the other 48.
- Colours: red, blue, green, yellow (Sol's four).
- Questions (Sol's three): "What color is {n}? Answer:", "Question: What is the color of {n}?
  Answer:", "Give only the color of {n}. Answer:".
- Fact sentence: "The color of {n} is {c}." Each sentence and each question is its own forward
  pass with an empty context, as in LiveMode.
- Site Q = the final token of a question. Site S = the name token.
- Four-colour accuracy = argmax over the four colour tokens is right. Strict = argmax over the
  whole vocabulary is right. Both are reported; gates use four-colour accuracy, like Sol's.

## G4.1 Address (zero parameters)

At every layer, score each candidate name m by cos(v̂_m, h_ℓ) and take the argmax. Report sites Q
and S, with the J-lens and with the logit lens (J = I) as comparison.
Read layer ℓ_r = the layer with the best dev accuracy at site Q (16 dev candidates); ties go to the
earlier layer.
**Pass:** test accuracy at site Q, layer ℓ_r, 48 candidates, ≥ 90% (chance 2.1%).

## G4.2 Write-back (zero parameters)

A question with no fact anywhere. The target colour c is each of the three colours that differ from
the model's own unprompted four-colour answer, so success requires a flip. Inject at site Q,
layer ℓ_w: h ← h + α‖h‖ v̂_c.
(ℓ_w, α) = best dev four-colour accuracy over all layers and α ∈ {0.5, 1, 2, 4, 8}; ties go to
the smaller α. Test: 48 names × 3 prompts × 3 targets = 432 trials.
Arms: J-lens vector, logit-lens vector (γ ⊙ u_c), a random unit direction, and no injection.
**Pass:** J four-colour ≥ 90% and random ≤ 40%.

## G4.3 Generalization: does the model reason with the injected concept?

Prompt: "Is {n} the same color as {obj}? Answer:". Objects: red = a ripe tomato, blue = a clear
summer sky, green = fresh spring grass, yellow = a ripe banana. Each test name gets a random
colour; half the items match and half do not. Answer = sign(logit " Yes" − logit " No"); score is
balanced accuracy.
Arms:
- ceiling: the fact "The color of {n} is {c}." in context;
- J injection of v̂_c at site Q with (ℓ_w, α) from G4.2;
- logit-lens injection;
- none.
**Pass:** J ≥ 75% and ceiling ≥ 75%. If the ceiling is below 75%, the gate is *uninformative*,
because the model cannot do the task even when told.
Prediction, not a pass condition: J beats the logit lens by ≥ 10 points.

## G4.4 Live native memory

Memory S ∈ R^{d×d}, starting at zero; keys and values live in the residual stream, restricted to the
J-space.
- **Write**, for each fact sentence:
  - key k = J-space component of h_{ℓ_r} at the token just before the value ("is");
  - value v = J-space component of h_{ℓ_w} at the value token;
  - S ← S + (v − S k) kᵀ, with k centred and unit-normalised and v unit-normalised.
  - The positions come from the sentence template, which is supplied structure. The identities do
    not: nothing tells the memory which name or colour it is writing.
- **Read**, for each question:
  - k_q = J-space component of h_{ℓ_r} at site Q, centred and normalised;
  - o = S k_q;
  - inject h_{ℓ_w}[Q] ← h + α‖h‖ o/‖o‖.
  - Two forward passes: the first captures k_q, the second injects.
- **Centring constants:** m_w = mean write-key J-component over 64 dev fact sentences;
  m_q = mean J-component at site Q over 48 dev questions. They are computed once from dev names and
  are not trained.
- **Episode:**
  - N test names get colours; then every name is overwritten once with a different colour,
    interleaved with 2N filler sentences;
  - each name is then queried once with a random question template;
  - the score is accuracy on the *latest* colour;
  - N ∈ {8, 16, 32}, 20 episodes each.
- **Arms:**
  - L, the live memory;
  - stale (memory after the first facts only);
  - none;
  - ctx (every sentence in one prompt, then the question: the in-context ceiling);
  - L_raw (uncentred keys, ablation);
  - oracle-key (key = v̂ of the name token at ℓ_r, both ways, a supplied address like Sol's hash);
  - wrong-address (read with another name's question: how often it returns the probed name's
    colour).
- **Pass:** L ≥ 80% at N = 8, L ≥ 60% at N = 32, and L − none ≥ 40 points at N = 8.

## What each outcome means

- G4.1 fails → Qwen-0.5B does not carry "who" in the J-space where the question is asked; a native
  address is not available at this scale, and the neuron route needs a different read site or a
  bigger model.
- G4.1 passes, G4.4 fails while oracle-key passes → the address exists but the memory cannot
  separate it from the shared content of the keys.
- G4.2 passes and G4.3 fails → the write makes the model *say* the word, but not think with it.
- All pass → a frozen LLM with zero trained parameters keeps an updatable memory in its own
  workspace. The only supplied structure is where the template puts the value.

Results are reported as registered, whatever they are.

## Addendum (2026-10-18, before any gate result; the J-lens was still computing)

After reading the paper's §4.3 more closely: it amplifies J-lens directions through the workspace
range with MLP gain up to about 10×, and it intervenes at one or more layers. A single-layer
injection may therefore wash out before a two-hop use such as yellow → banana in G4.3. One arm is
added to G4.2 and G4.3, **reported only, with no pass criterion, and none of the registered gates or
selections change**:

- **J_band:** inject the colour's own J-lens vector at each of the four layers ending at ℓ_w
  (ℓ_w−3 … ℓ_w). Each layer gets α/4 · ‖h_ℓ‖ at site Q, with α taken from G4.2.

Possible next gate, not part of this run: restrict reads and writes to the subspace of the
"broadcast heads". These are the top 1% of workspace-layer heads by OV gain and label preservation
on J-lens vectors, computable from the weights alone (paper §4.3.2).
