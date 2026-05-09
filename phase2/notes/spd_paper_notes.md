# SPD (Stochastic Parameter Decomposition) — Operational Reference

## 1. Decomposition setup

SPD decomposes each weight matrix W^l (one per layer, l=1..L) of a frozen target
model into C **rank-one subcomponents**:

    W^l_{i,j} ≈ Σ_{c=1}^C  U^l_{i,c} · V^l_{c,j}

So each subcomponent is the outer product u^l_c · v^l_c^T, NOT a full-shape copy
of W^l. Storage per subcomponent is O(d_in + d_out), not O(d_in · d_out) as in
APD's parameter components. C may exceed min(d_in, d_out) (lets it find
superposed computations). The subcomponents contribute by being summed (with
optional masking) to produce the layer's effective weight matrix during a
forward pass. Subcomponents living in different layers are later clustered into
"full parameter components"; in this paper that clustering is implicit/manual.

## 2. Causal importance functions

For each layer l and subcomponent c there is a tiny per-subcomponent MLP γ^l_c
that takes a *scalar* input — the subcomponent's own "inner activation"
h^l_c(x) = Σ_j V^l_{c,j} · a^l_j(x) (i.e. the activation entering W^l projected
onto v^l_c). Architecture: 1 hidden layer, GELU, width d_gate (16 typical, 128
for the deepest model), scalar in / scalar out.

    g^l_c(x) = σ_H( γ^l_c( h^l_c(x) ) )    in [0,1]

σ_H is a hard sigmoid; in practice the paper uses *leaky* hard sigmoids:
lower-leaky (slope 0.01 below 0) for the forward-pass gates, upper-leaky (slope
0.01 above 1) inside the importance-minimality penalty.

g^l_c(x) is the model's prediction of "how non-ablatable subcomponent c is on
input x". g≈0 → fully ablatable (causally unimportant); g≈1 → must remain
unmasked.

## 3. Stochastic masks

Per training step, sample r^l_c ~ Uniform(0,1) and form the **mask**

    m^l_c(x, r) = g^l_c(x) + (1 − g^l_c(x)) · r^l_c

This is exactly the `mask = ci + (1 − ci) * source` formula in CLAUDE.md.
Equivalently m^l_c ~ Uniform(g^l_c(x), 1). Important consequences:

- If g=1, mask ≡ 1 (no ablation).
- If g=0, mask ~ U(0,1) (any ablation is fair game).
- It is NOT a Bernoulli; masks are continuous scalars.
- Reparametrization through r lets gradients flow into γ^l_c.

The masked weight at a layer is W'^l(x,r) = U^l · diag(m^l(x,r)) · V^l. S mask
samples per step (S=1 sufficed in all experiments).

## 4. The losses

Total: L_SPD = L_faith + β1·L_recon + β2·L_recon-layerwise + β3·L_imp-min

(a) **Faithfulness** — sum of subcomponents must equal W (parameter-space, no
data needed):

    L_faith = (1/N) Σ_l Σ_{i,j} ( W^l_{i,j} − Σ_c U^l_{i,c} V^l_{c,j} )^2

(b) **Stochastic reconstruction** — masked model output ≈ target output:

    L_recon = (1/S) Σ_s D( f(x | W'(x, r^(s))) , f(x | W) )

where D is KL for LMs, MSE for toys.

(c) **Layerwise stochastic reconstruction** — same but mask only one layer at a
time (helps reduce gradient noise):

    L_recon-layerwise = (1/(L·S)) Σ_l Σ_s D( f(x | …, W'^l, …) , f(x | W) )

(d) **Importance minimality** (this is the sparsity term):

    L_imp-min = Σ_l Σ_c | g^l_c(x) |^p ,   p > 0

p=1 or p=2 in experiments. Pushes g down so as many subcomponents as possible
become ablatable; without it the trivial g≡1 solution is optimal. There is NO
simplicity / Schatten-norm loss (a deliberate change from APD).

## 5. APD → SPD changes

| | APD | SPD |
|---|---|---|
| Component shape | full-shape vector in entire parameter space | per-layer rank-one subcomponents (U·V^T) |
| Activation selection | gradient attribution + top-k | learned causal importance + stochastic masks |
| Sparsity hyperparam | top-k (very brittle) | β3 importance penalty (forgiving) |
| Gradient flow | only through top-k components | through all subcomponents every step |
| Simplicity loss | Schatten-p norm on singular values | none (rank-1 by construction) |
| Shrinkage | ML2R ≈ 0.9 (significant) | ML2R ≈ 1.0 |
| Cross-layer components | each component spans all L layers | found by post-hoc clustering of subcomponents |

The combined effect — no top-k, no simplicity loss, dense gradient flow,
learned (not attributed) importance — is what makes SPD scale and stop needing
careful tuning.

## 6. CLAUDE.md terminology mapping

- **Sources** (`adv_sources`, `self.sources`, `PPGDSources`) → r^l_c in the
  paper. Raw scalars in [0,1]. In standard SPD they are sampled iid uniform; in
  this repo's PGD machinery they can additionally be optimized adversarially to
  find worst-case ablation patterns.
- **Masks** (`component_masks`, `RoutingMasks`, `make_mask_infos`) → m^l_c in
  the paper, materialized via `mask = ci + (1 − ci) * source`. ci is g^l_c(x).
- **n_mask_samples** → S in the paper (mask samples per training step;
  paper-default 1).
- **RoutingMasks / MaskInfos** → bookkeeping for which subcomponents get masked
  on which forward pass (per-layer m^l vectors plus per-batch metadata).

So `mask = ci + (1-ci)*source` is literally the SPD interpolation. CI = causal
importance = g^l_c(x). Source = r^l_c. Mask = m^l_c.
