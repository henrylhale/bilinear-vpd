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

## Per-SUBJ component structure: VPD recovers verb-equivalence classes

`phase2/analyze_per_subj.py` runs the same forward pass but conditions on the specific `tokens[t-1]` SUBJ token at every bigram-firing position. The output (`runs/vpd_v16_B/per_subj_analysis.txt`) gives, per-module, the mean importance of every component when each of the 8 SUBJ tokens is the trigger.

**The decomposition's bigram-handling components in Layer 0 attention cluster by *shared top verb*, not by raw subject identity.** From the DGP rule table:

| top-verb | SUBJ tokens that map to it |
|---|---|
| VERB_1 | SUBJ_6 |
| VERB_3 | SUBJ_7 |
| VERB_6 | SUBJ_2 |
| **VERB_7** | **SUBJ_0, SUBJ_3** |
| **VERB_8** | **SUBJ_4, SUBJ_5** |
| VERB_11 | SUBJ_1 |

There are exactly two non-trivial equivalence classes (`{SUBJ_0, SUBJ_3}` and `{SUBJ_4, SUBJ_5}`). The matching `Layer 0 K1` components are exactly these clusters:

| component | per-SUBJ mean importance |
|---|---|
| `blocks.0.attn.k1_proj:10` | S0=0.69, **S3=0.72**, others ≈ 0 |
| `blocks.0.attn.k1_proj:29` | S0=0.67, **S3=0.70**, others ≈ 0 |
| `blocks.0.attn.k1_proj:9`  | S0=0.58, **S3=0.61**, S7=0.03, rest=0 |
| `blocks.0.attn.k1_proj:5`  | S4=0.72, **S5=0.71**, S6=0.20, rest=0 |

A handful of components fire for **exactly** the SUBJ pairs that share a top verb. For the six subjects with unique top-verbs there are corresponding singleton components (`q2_proj:26` for SUBJ_1, `k2_proj:11` for SUBJ_2, etc.).

The decomposition has compressed redundant rules: rather than 8 components-per-SUBJ, it found 6 components (one per top-verb), with the two doubly-mapped verbs picked out by SUBJ-pair components. This is exactly the "merge equivalent computations" behavior VPD is supposed to produce.

By contrast, Layer 1 MLP components (`w_m`, `w_n`) show much weaker per-SUBJ structure — their dominant components fire on almost all SUBJ types at similar strength. Those are the **shared "verb-prediction" machinery** that runs regardless of which subject triggered the bigram. The split is consistent with the natural architecture: layer 0 attention identifies *which verb-class*, layer 1 MLP completes the lookup.

## Per-trigger-slot structure of induction

`phase2/analyze_induction.py` buckets induction-firing positions by the slot of the trigger token (`tokens[t-1]`) — VERB / LOC / ADJ / CONN (SUBJ is impossible since bigram primitive takes precedence whenever prev=SUBJ).

**Layer 1 attention keys have dedicated per-slot components:**

```
=== blocks.1.attn.k1_proj  C=32  (selectivity = max/min across trigger slots) ===
  comp   VERB    LOC    ADJ   CONN   sel
     7   0.70   0.11   0.15   0.46   6.54   <- VERB trigger
    18   0.39   0.80   0.17   0.57   4.62   <- LOC trigger
     3   0.51   0.16   0.37   0.65   4.06   <- CONN trigger
     5   0.17   0.42   0.24   0.11   3.83   <- LOC trigger
     8   0.22   0.16   0.59   0.40   3.63   <- ADJ trigger
    14   0.27   0.16   0.56   0.37   3.40   <- ADJ trigger
    28   0.17   0.50   0.31   0.15   3.33   <- LOC trigger
    26   0.12   0.35   0.37   0.36   3.23   <- ADJ/CONN
```

Each non-SUBJ slot has at least one dedicated component (often two). The structure is exactly the right one for an induction-head implementation: Layer 1 keys say "I'm a position whose previous token is in slot X" so that Layer 1 queries can match against them. Different slot ⇒ different key direction ⇒ different component.

Layer 1 V (`blocks.1.attn.v_proj`) shows weaker slot-selectivity (max sel ≈ 2), concentrated on ADJ — the value vectors are doing more shared work. Layer 1 MLP and unembed show very weak slot-structure (selectivity ~1.5), confirming they implement *primitive-agnostic* completion machinery.

## Per-rule structure of skip-trigram

`phase2/analyze_skip.py` conditions on the specific `(LOC, ADJ)` pair that triggered each skip firing. Layer 0 attention has *per-rule* dedicated components:

```
=== blocks.0.attn.k1_proj  C=32  (sel = max/mean-others across rules) ===
  comp  dominant rule           mean   others_avg   sel
    24  L5+A2 -> C7             0.46   0.040       11.5   <- single-rule
    22  L1+A4 -> C1             0.11   0.009       11.5
    16  L7+A7 -> C0             0.24   0.021       11.5
    26  L3+A0 -> C1             0.08   0.007       11.5
    12  L3+A0 -> C1             0.60   0.052       11.5  (same rule as comp 26)
    17  L7+A7 -> C0             0.43   0.038       11.5  (same rule as comp 16)
     7  L0+A1 -> C5             0.46   0.091        5.1
    15  L5+A4 -> C6             1.01   0.239        4.2
```

The `sel = 11.5` ceiling is an artifact of the metric (mean-of-others denominator gets small for rare rules), not a saturation in the actual data — individual components really do fire on essentially one rule out of 24, with mean importance ≈ 0 elsewhere.

Multiple components often fire on the *same* rule (e.g. comps 26 + 12 both implement `(LOC_3, ADJ_0) → CONN_1`). This is mild redundancy / lack of compression; subset routing during training might pressure these into a single shared component. Worth checking after a subset-routing retrain.

Layer 0 V is similarly rule-selective (sel ≈ 2-6); Layer 0 O is much more shared (sel ≈ 1.5-2.5) — that's the projection that consolidates the rule's effect back into the residual stream, so it doesn't need to be rule-specific.

Layer 1 MLP w_m and unembed both show per-rule structure but at lower selectivity (sel ≈ 2-5) — they're computing "which CONN token to emit" and that's already shared across multiple input rules that emit the same CONN.

## Summary so far

The decomposition has cleanly recovered:

| primitive | mechanism                                                       | components live in              |
|-----------|------------------------------------------------------------------|---------------------------------|
| bigram     | 6 per-top-verb classifiers (compressed from 8 SUBJ rules)        | Layer 0 K1/K2/Q1/Q2, ~3 per class |
| skip-trigram | 24 per-rule detectors                                          | Layer 0 K1, ~1-2 per rule       |
| induction  | 4 per-slot Layer-1 K detectors (1-2 components per VERB/LOC/ADJ/CONN trigger slot) | Layer 1 K1                    |
| filler     | shared default machinery                                         | embed, unembed, Layer 1 MLP    |

This matches the architectural intuition: Layer 0 attention identifies the *trigger* of bigram and skip-trigram primitives (which depend only on the immediate previous token + recent context), Layer 1 attention runs the induction matching, and the MLP/unembed do shared completion work.

## Open questions / what's next

- **Subset-routing retrain.** Confirm the per-rule findings are stable when training penalizes co-firing of redundant components (paper's `UniformKSubsetRouter`). Should compress the multi-component-per-skip-rule cases (`L3+A0→C1` had 4-5 dedicated components, probably should collapse to 1).
- **PPGD.** Adversarial mask sampling would tell us whether the current decomposition's faithfulness is robust to adversarial subset selection or just to the typical training subset.
- **Geometric verification.** Take the bigram-class-cluster components from Layer 0 K1 (e.g. comp 10 = `{SUBJ_0, SUBJ_3} → VERB_7`) and check that their V column in embed-component space aligns with the SUBJ_0 + SUBJ_3 embedding directions; same for Layer 1 K slot components and the relevant slot's embedding-component projection. This closes the loop from "this component fires when X" to "this component computes inner-product-with-X".
- **Interaction-node graph.** The actual deliverable from the VPD paper: build a graph where nodes are (component, position) pairs and edges are attribution scores `(∂a_c / ∂a_c')* · a_c' · g_c'`. Useful for visualizing how, e.g., a Layer 0 K1 SUBJ-class component flows downstream to specific Layer 1 MLP w_m components.
