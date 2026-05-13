# Phase 2 — VPD decomposition findings

First serious analysis of the VPD decomposition `runs/vpd_v16_B/`.

## Decomposition quality

End of 200k-step VPD training with annealed `coeff_imp` (0 → 1.6e-5 over steps 10k–100k, then constant):

- `recon_kl = 0.022` (KL between target v16 logits and component-model masked logits)
- `ci_l0_frac = 0.310` (fraction of components with importance ≥ 0.5, averaged across batches and the 20 wrapped modules)
- The decomposition is faithful (recon nearly zero) AND meaningfully sparse (~31% of components fire per input).

The earlier degenerate-dense Run A (`coeff_imp = 0` constant) confirms the decomposition has enough capacity to fit v16 exactly (recon = 0, l0 = 1.0); Run B finds the Pareto-knee.

## Per-module average CI on each primitive's positions

Measured by averaging `ci_upper_leaky` across 1024 sequences (256-sequence batches × 4) annotated by which primitive fires at each position. Numbers below are *per-module mean* across components.

```
module                         #C    bigram     skip   induct   filler
blocks.0.attn.k1_proj          32     0.484    0.428    0.225    0.212
blocks.0.attn.k2_proj          32     0.468    0.379    0.187    0.173
blocks.0.attn.o_proj           32     0.668    0.827    0.682    0.422
blocks.0.attn.q1_proj          32     0.447    0.387    0.421    0.416
blocks.0.attn.q2_proj          32     0.430    0.433    0.407    0.398
blocks.0.attn.v_proj           32     0.624    0.619    0.298    0.277
blocks.0.mlp.w_m               64     0.199    0.233    0.210    0.155
blocks.0.mlp.w_n               64     0.177    0.209    0.202    0.148
blocks.0.mlp.w_proj            64     0.212    0.265    0.241    0.175
blocks.1.attn.k1_proj          32     0.116    0.324    0.401    0.237
blocks.1.attn.k2_proj          32     0.047    0.315    0.470    0.267
blocks.1.attn.o_proj           32     0.273    0.233    0.251    0.094
blocks.1.attn.q1_proj          32     0.396    0.399    0.270    0.258
blocks.1.attn.q2_proj          32     0.460    0.399    0.303    0.292
blocks.1.attn.v_proj           32     0.607    0.677    0.517    0.491
blocks.1.mlp.w_m               64     0.373    0.341    0.226    0.162
blocks.1.mlp.w_n               64     0.378    0.344    0.224    0.166
blocks.1.mlp.w_proj            64     0.221    0.206    0.276    0.166
embed                          64     0.787    0.736    0.732    0.724
unembed                        64     0.608    0.574    0.619    0.596
```

## Mechanistic reads

### Bigram + skip-trigram are handled in layer 0
- Layer-0 K/Q/V projections are most active on bigram and skip positions (`k1_proj`, `k2_proj`, `v_proj` ≈ 0.4–0.6 for bigram/skip vs 0.17–0.30 for induction/filler).
- The `o_proj` shows a *skip-spike* (0.83 vs filler 0.42, induction 0.68). Skip-trigram requires the "look at recent LOC" lookback that Layer 0 attention does explicitly.

### Layer 1 attention keys do the induction lookup
- `blocks.1.attn.k1_proj` and `k2_proj` are *suppressed* on bigram positions (0.12, 0.05) and *boosted* on induction positions (0.40, 0.47).
- This is exactly the expected role for the induction head: Layer 1 keys encode "I am position s+1; my previous token was X". When the current query token equals X (the induction trigger), the layer-1 attention concentrates on these keys.
- The original selectivity-vs-filler metric missed this because Layer-1 keys still fire moderately on filler positions; the discriminating signal is *between primitive types*, not between primitive and filler.

### Layer 1 MLPs split bigram and induction
- `blocks.1.mlp.w_m` and `w_n`: ~0.37 on bigram positions, ~0.22 on induction. The MLP has separate "bigram completion" and "induction completion" pathways — components in Layer 1 MLP fire heavily on bigram positions (where the previous token is SUBJ and we're predicting a verb) and less on induction.
- `blocks.1.mlp.w_proj`: roughly equal across primitives, suggesting the final projection mixes both back into the residual stream.

### Embed/unembed are essentially primitive-agnostic
- Embed: 0.78/0.74/0.73/0.72 across bigram/skip/induct/filler — every input token uses the embedding, regardless of which primitive is firing.
- Unembed: 0.61/0.57/0.62/0.60 — same, every output prediction uses the unembed.
- Components in embed/unembed represent the *typed vocabulary structure* (SUBJ vs VERB vs etc.), not the rule logic.

## Strong selectively-firing components (selectivity > 3× filler)

Top per-module specialists (from `runs/vpd_v16_B/component_analysis.txt`):

- `blocks.0.attn.k1_proj`: component 7 — bigram, selectivity 9.8×
- `blocks.0.attn.q1_proj`: component 10 — bigram, **selectivity 37×** (essentially never fires outside bigram positions)
- `blocks.0.mlp.w_proj`: component 24 — skip, selectivity 7.4×
- `blocks.1.mlp.w_m`: 8 components specialized for bigram + 8 for skip
- `unembed`: components 20, 17 — induction, selectivity ~1.9× (weak but the only place where any component shows up as induction-preferring on the filler-baseline metric)

## Open questions / what's next

- **Which specific bigram-component corresponds to which SUBJ token?** The DGP has 8 SUBJ tokens with distinct verb conditional distributions. A clean decomposition should have 8 dedicated bigram-component clusters, one per subject. To check, condition the analysis on `tokens[t-1]` (the trigger SUBJ token) and look for per-SUBJ component firings.
- **Which Layer 1 key components implement the induction match?** Pick the top Layer-1 K components by induction/bigram ratio (e.g. k2_proj components with induct=0.47, bigram=0.05) and look at their cosine similarity to the embed-component subspace — they should align with token-identity directions.
- **Subset-routing retrain.** The current decomposition was trained with `AllLayersRouter` (mask every layer always). The paper's subset routing should disambiguate components that currently co-fire as a pair.
- **PPGD.** Adversarial mask sampling would tell us whether the current decomposition's faithfulness is robust to adversarial subset selection or just to the typical training subset.
