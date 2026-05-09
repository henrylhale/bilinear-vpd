# VPD Reference Notes — adVersarial Parameter Decomposition

Bushnaq et al., Goodfire, May 2026, *Interpreting Language Model Parameters*. Equations cross-checked against `goodfire-ai/param-decomp` reference impl `nano_param_decomp/run.py`. VPD = SPD (Stochastic Parameter Decomposition, arXiv 2506.20790) + persistent adversarial PGD masks.

## 1. Decomposition

For each decomposed `nn.Linear` weight $W^l\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$:

$$W^l \;=\; \sum_{c=1}^{C^l} \vec{U}^l_c (\vec{V}^l_c)^\top \;+\; \Delta^l,\qquad \Delta^l := W^l - \sum_c \vec{U}^l_c(\vec{V}^l_c)^\top\quad\text{(Eq. 1)}$$

A "component" is one rank-1 outer product. $\Delta^l$ is a per-position spillover with its own scalar mask $m_\Delta\in[0,1]$. **Init:** $V\sim\mathcal{N}(0,1/d_\text{in})$, $U\sim\mathcal{N}(0,1/C)$. Original $W^l$/bias frozen.

**Causal-importance function** $g_\theta$: a single shared bidirectional transformer (RoPE Q/K) seeing pre-weight activations of *all* decomposed modules concatenated. Outputs $g^l_{b,t,c}\in[0,1]$ via leaky-hard sigmoids. Two heads: `ci_lower` (clamp $[0,1]$, leaky negative grad — resurrection of dead components) for masks; `ci_upper` (linear continuation $1+\alpha(x-1)$ above 1) for the minimality loss.

**Masks:** $m^l_{b,t,c}\in[g^l_{b,t,c},1]$, parameterized as $m=g+(1-g)s$.
- **Stochastic:** $s\sim U(0,1)$. Delta mask $\sim U(0,1)$.
- **Adversarial (PGD):** $s$ is a persistent per-(batch,pos) source trained to *maximize* recon KL.

**Routing (stochastic only):** each position picks $k\sim\mathrm{Uniform}\{1..M\}$ random modules to actually replace; un-routed pass through original $W^l$. PPGD routes all layers. This subset routing is what enforces "ablatable in any combination, not just jointly" (§3.5) and prevents feature-splitting.

## 2. Losses

$$\mathcal{L}_\text{VPD}=\beta_\text{faith}\mathcal{L}_\text{faith}+\beta_\text{imp}\mathcal{L}_\text{imp}+\beta_\text{stoch}\mathcal{L}_\text{stoch}+\beta_\text{ppgd}\mathcal{L}_\text{ppgd}$$

**Faithfulness** ($\Delta\to 0$ so components sum to $W$):
$$\mathcal{L}_\text{faith}=\frac{\sum_l\|\Delta^l\|_F^2}{\sum_l\mathrm{numel}(\Delta^l)}$$
Driven hard by $\beta_\text{faith}=10^7$. **400 dedicated warmup steps at lr=1e-3 before main loop** (mandatory).

**Reconstruction** (output preserved under masks; KL is target→prediction):
$$\mathcal{L}_\text{recon}(m)=\mathbb{E}_x\big[D_\text{KL}\big(f(x|W)\,\|\,f(x|W'(m))\big)\big]\quad\text{(Eq. 3)}$$
where $W'(m)$ replaces each $W^l$ with $V^l\mathrm{diag}(m^l_{b,t,\cdot})(U^l)^\top + m_\Delta\Delta^l$.

**Minimality + description length** (Eqs. 2 & 4):
$$\mathcal{L}_\text{imp}=\sum_{l,c}\Big[\overline{(g^l_c)^p}+\beta_\text{desc}\overline{(g^l_c)^p}\log_2\!\big(1+\textstyle\sum_{b,t}(g^l_{b,t,c})^p\cdot W\big)\Big]$$
$\overline\cdot$ is per-component mean over batch/seq; $W$=DDP world size; $\beta_\text{desc}=0.5$. **$p$ anneals 2.0→0.4 linearly.** Log term penalizes components that fire on many positions (frequency sparsity, on top of per-position sparsity).

**Adversarial PGD inner loop** (`per_batch_per_position` scope): source $s\in[0,1]^{B\times S\times(C+1)}$ persists across outer steps with its own Adam state $(m,v,t)$ and $(\beta_1,\beta_2)=(0.5,0.99)$, $\epsilon=10^{-8}$. Each outer step: run **2 PGD inner steps** maximizing $\mathcal{L}_\text{ppgd}$, then take outer V/U/$g$ gradient that minimizes the same. Source grads extracted via `torch.autograd.grad` *before* `total.backward()` (so adversary doesn't pollute V/U grads); source Adam step happens *after* `opt.step()`. Projection: `src.clamp_(0,1)`. **Eval PGD is different**: 20 sign-SGD steps, step_size 0.1, one source shared across batch.

## 3. Hyperparameters (paper 4L vs SimpleStories-2L)

|  | Pile-4L (paper) | SS-2L | Notes |
|---|---|---|---|
| Target | 4-layer LlamaSimpleMLP, $d=768$, 6 heads, 3072 MLP, RoPE/GELU | 2-layer same arch | "MLP-only Llama" = standard Llama, attn intact |
| Decomposed mods | q/k/v/o/c_fc/down_proj × 4 = 24 | × 2 = 12 | bias not decomposed |
| $C$ per module | 512/512/1024/1024/3072/3584 | 288/288/384/480/1152/960 | ~$4\times$ module dim |
| Total comps | 38,912 (~10K alive at >1e-6) | 9,408 | |
| n_steps / batch / seq | 400k / 64 / 512 | same | |
| Optim outer | AdamW, wd=0 | same | |
| `main_lr` cosine | 5e-5 → 5e-6 | **3e-4 → 3e-5** | tinier model wants higher lr |
| Faithfulness warmup | 400 steps lr=1e-3 (V,U only) | same | |
| $\beta_\text{faith}$ | 1e7 | 1e7 | |
| $\beta_\text{imp}$ | **2e-4** | **1e-3** | **primary sparsity knob** |
| $\beta_\text{stoch}$ / $\beta_\text{ppgd}$ | 0.5 / 0.5 | 0.5 / 0.5 | |
| $p$ anneal / $\beta_\text{desc}$ / $\epsilon$ | 2.0→0.4 / 0.5 / 1e-12 | same | |
| `leaky_alpha` | 0.01 | 0.01 | |
| PPGD: inner_steps / lr / warmup% / Adam | 2 / 0.01 / 2.5% / (0.5,0.99) | same | |
| Eval PGD | 20 sign-SGD, step 0.1 | same | |
| `grad_clip_components` | **0.01** | 0.01 | only on V,U, not CI net |
| CI transformer | $d_m$=2048, 8 blocks, 16 heads, MLP=8192 | 512 / 4 / 8 / 2048 | RoPE base 1e4, RMSNorm, bidir |

**Per-layer alive (4L Table 2):** L0=3709, L1=848, L2=1943, L3=3472. **Mean $L_0$/position:** 44.6/18.9/49.5/92.0, total ~205 (~2.1% of alive fire per token).

## 4. Interaction-node graph (§5)

**Nodes:** causally-important $(c, t)$ pairs (subcomponent × position), filtered by $g_{c,t}>\tau$.

**Edge attribution** (§5.1):
$$\mathrm{attr}(c'\to c) = (\partial a_c/\partial a_{c'})^{*}\cdot a_{c'}\cdot g_{c'}$$
$a_c = (V^\top x)_c$. $(\cdot)^*$ = stop-gradient on every subcomponent except the source $c'$, isolating the *direct* path through $c'$.

**Pruning** (full graph intractable):
1. **Adversarial mask sampling** — fit a fresh minimal mask requiring CE-on-label stays good; graph over those components.
2. **Top-k attribution** from the unmasked forward.

**Attention-head edges** (§§4.3-4.4):
- Static: $(\mathrm{sign}\,\mathbb{E}_\phi[\phi V_{Q,c}]\,\|V_{Q,c}\|U^h_{Q,c})^\top R_\tau\,(\mathrm{sign}\,\mathbb{E}_\phi[\phi V_{K,c'}]\,\|V_{K,c'}\|U^h_{K,c'})$
- Data-dependent: $(\phi V_{Q,c}(U^h_{Q,c})^\top R_\tau U^h_{K,c'}V_{K,c'}^\top\phi^\top)_{t,t'}$

$R_\tau$ = relative-position RoPE rotation. **§7.2 caveat:** these are *attribution* graphs, not computational graphs — they linearize past nonlinearities; a scalar edge can hide a switch.

## 5. Gotchas for a tiny 2-layer bilinear

1. **Bilinear ≠ Llama-MLP.** Decompose each of the two bilinear projections separately as `ComponentLinear`. CI net's input is the *pre-weight activation* of each wrapped linear (the two branches likely share input tensor — fine).
2. **Right-size $C$.** Paper ratio is ~4×rank, ~25% alive. For an 80K-param toy, total $C$ in the low hundreds is plenty; over-provisioning wastes training.
3. **`coeff_imp` is THE knob.** Sweep $\{1e\!-\!4, 3e\!-\!4, 1e\!-\!3, 3e\!-\!3\}$, plot KL-vs-$L_0$ Pareto. Too high → zero solution (KL explodes, $L_0\to 0$).
4. **Faithfulness warmup is mandatory.** Verify $\|\Delta\|_F^2/\mathrm{numel}\lesssim 10^{-8}$ before main loop.
5. **`grad_clip=0.01` on V,U only** (not CI). Tight clip; keep it.
6. **CI net can be tiny.** $d_m\!=\!128\text{-}256$, 2 blocks, 4 heads is enough for an 80K target.
7. **Don't break subset routing.** Forcing all-layers routing for stochastic recon causes SAE-like over-splitting. Keep uniform-$k$.
8. **`leaky_alpha=0.01` enables resurrection.** Plain `clamp` strands dead components forever.
9. **KL direction is fixed:** target=ref (detached), components=q. Reversing destabilizes.
10. **Bias frozen, not decomposed.** If your bilinear has biases, they pass through unmasked.
11. **Don't chase complete adversarial robustness** (§7.3). Paper's KL stays sane to ~80 PGD steps then explodes. Pick honest step count (20 with step 0.1) and report consistently.
12. **Interaction graphs are unreadable on toy data.** Inspect top-attribution edges per node manually instead of rendering a graph.
13. **Description-length log term has `*world_size` baked in.** No-op single-GPU; don't manually correct.
14. **Source-grad order matters in training step:** `autograd.grad(loss_ppgd, sources, retain_graph=True)` *before* `total.backward()`, then `opt.step()`, then `ppgd.external_step(grads, lr)`. Reordering corrupts gradients.
15. **Attention decomposition is the novelty over SAEs/transcoders.** A bilinear-MLP-only target uses ~half of VPD; the MLP path is essentially SPD (still useful).
