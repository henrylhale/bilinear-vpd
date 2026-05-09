# nano_param_decomp/run.py — operational reference for one PD step

Single-file Stochastic/Visual Parameter Decomposition (SPD/VPD) implementation. Source: `/home/henry/bilinear-vpd/nano_param_decomp/run.py` (1219 lines), entry points `simplestories_2L.py` and `pile_4L.py`.

## 1. High-level structure (the file's own labels)

- **A. Config** (lines 74–150): the `Config` dataclass.
- **B. Leaky-hard sigmoids** (153–195): `lower_leaky` (custom autograd), `upper_leaky`.
- **C. ComponentLinear + install helper** (198–283): wrapper module replacing each `nn.Linear`, plus `install_components`.
- **D. CI transformer** (286–398): `precompute_rope`, `apply_rope`, `CIAttention`, `CIBlock`, `CITransformer` — the shared causal-importance net.
- **E. Losses + mask/routing sampling** (401–512): `faithfulness_loss`, `importance_minimality_loss`, `kl_logits`, `sample_continuous_masks`, `sample_uniform_k_subset_routing`, `set/clear_wrapper_masks`, `stochastic_recon_loss`.
- **F. Persistent PGD** (515–608): `PersistentPGD` class.
- **G. LR schedule** (611–628): `cosine_lr`.
- **H. Distributed setup + SPDModule** (631–680): `init_dist`, `SPDModule`.
- **I. Training loop** (683–846): `decompose` — faithfulness warmup, then main loop.
- **J. Eval metrics** (849–1219): `eval_ci_l0`, `eval_ce_kl_losses`, `eval_pgd_recon`, `eval_hidden_acts_recon`, etc.

## 2. Wrapping a target model

`install_components(model, module_to_c)` (line 270):
- freezes every parameter of the target,
- iterates `cfg.C_per_module: dict[str, int]` mapping submodule paths → component count,
- uses `model.get_submodule(parent_path)` + `setattr` to swap each `nn.Linear` for `ComponentLinear(orig_linear, C)`,
- returns `dict[str, ComponentLinear]`.

**Submodule paths** are dotted names like `h.0.attn.q_proj`, `h.0.mlp.c_fc`, `h.0.mlp.down_proj` (see `pile_4L.py` lines 29–54 and `simplestories_2L.py` lines 35–48). Six module types per layer: `attn.{q,k,v,o}_proj`, `mlp.c_fc`, `mlp.down_proj`.

`ComponentLinear` (line 201):
- stores the original `nn.Linear.weight` as a frozen buffer `W_target` `[d_out, d_in]` plus optional `bias`,
- adds trainable parameters `V: [d_in, C]` and `U: [C, d_out]`, init `N(0, 1/sqrt(d_in))` and `N(0, 1/sqrt(C))`,
- has a "weight delta" `delta = W_target - (V @ U).T` — the residual not yet captured by VU,
- two forward modes:
  - `"target"`: caches `x` into `last_input` (used by the CI transformer) and returns `F.linear(x, W_target, bias)`.
  - `"component"`: with externally-set `mask: [B,S,C]`, `delta_mask: [B,S]`, optional `routing_mask: [B,S] bool`:
    - `comp_acts = x @ V`
    - `comp_out = (comp_acts * mask) @ U + bias + delta_mask[...,None] * (x @ delta.T)`
    - if `routing_mask` is set, mix per-position with the target output (`torch.where`).

## 3. Forward pass under a mask sample

For one training step on a token batch `input_ids: [B, S]`:

1. `SPDModule.forward(input_ids)`: clears wrapper masks; runs `target_model(input_ids)` in `"target"` mode → `target_logits` and each wrapper's `last_input` cached.
2. `CITransformer(acts)`: RMS-norm each cached input, concat along feature dim in alphabetical module-path order, project to `d_model`, run `n_blocks` bidirectional attention+MLP blocks (RoPE on Q/K, no causal mask), project to `total_C`, split per module, apply `lower_leaky` and `upper_leaky` → `ci_lower`, `ci_upper`.
3. Sample masks per module: `mask = ci_lower + (1 - ci_lower) * U(0,1)`, `delta_mask ~ U(0,1)` per `(b,s)`.
4. Sample uniform-k-subset routing per `(b,s)`: pick `k ~ Uniform{1..M}` (M = #modules), route a random k-subset to component mode, the rest to target mode.
5. Set wrapper masks; run `target_model(input_ids)` again to get pred logits; clear wrapper masks.
6. Compute losses (next section).

## 4. Loss terms (exact)

All four losses sum into `total = c_faith*L_faith + c_imp*L_imp + c_stoch*L_stoch + c_ppgd*L_ppgd` (line 790).

**Faithfulness** (line 404):
```
L_faith = sum_m sum_ij (W_target_m - (V_m U_m).T)_ij^2 / total_numel
```

**Reconstruction (KL on logits)** (line 439): `kl_logits(pred, target)`:
```
L_recon = mean_{b,s} sum_v softmax(target)_v * (log_softmax(target) - log_softmax(pred))_v
```
i.e. `KL(softmax(target.detach()) || softmax(pred))` averaged over (B,S). Used by both `stochastic_recon_loss` (one fresh mask sample with k-subset routing) and `ppgd.recon_loss` (current persistent sources, all-layers routing).

**Importance minimality** (line 420), per-module on `ci_upper`:
```
For each module v of shape [B,S,C]:
  vals = (v + eps)^p                          # p anneals linearly p_start -> p_end
  sum_c = sum over (B,S) of vals              # [C]
  mean_c = sum_c / (B*S)
  contrib = sum_c [ mean_c + beta * mean_c * log2(1 + sum_c * world_size) ]
L_imp = sum over modules of contrib
```
With p_end < 1 this is an approximate L_p sparsity; the log term penalises components alive globally.

**PGD recon** (line 564) — same KL form, but masks come from persistent adversarial sources:
```
mask = ci_lower + (1 - ci_lower) * sources[..., :C]
delta_mask = sources[..., -1]
L_ppgd = kl_logits(pred, target_logits)
```

There is no auxiliary "identity insertion" or extra reconstruction term. The weight delta acts as a "spillover" component (an extra C+1th slot fed by `delta_mask`).

## 5. PGD (PersistentPGD, section F)

Per-rank, per-(batch,position) adversarial sources of shape `[local_B, S, C+1]` per module, `clamp(0,1)`. Adam state (m, v, t) is persisted across training steps — sources are NOT re-initialised each step.

Per training step:
1. `ppgd.warmup(...)` runs `cfg.ppgd_inner_steps = 2` PGD updates *before* loss computation:
   - forward target with `mask = ci + (1-ci)*src[...,:C]`, `delta_mask = src[...,-1]`, all-layers routed (no `routing_mask`),
   - `loss = kl_logits(pred, target_logits)`,
   - `grads = torch.autograd.grad(loss, sources)`,
   - Adam update: `src += lr * (m/bc1) / (sqrt(v/bc2) + eps)` (ascends loss — note `+=`, not `-=`),
   - `clamp_(0, 1)`.
2. `ppgd.recon_loss(...)` runs forward again with current sources → produces `loss_ppgd` for the main backward.
3. Before `total.backward()`, `torch.autograd.grad(loss_ppgd, sources, retain_graph=True)` extracts source grads.
4. After `total.backward()` and `opt.step()`, `ppgd.external_step(...)` Adam-updates sources from those grads (one more PGD step using the same persistent state).

PGD lr follows its own cosine schedule with `ppgd_warmup_pct = 0.025`, `beta1=0.5`, `beta2=0.99`. PGD eval (`eval_pgd_recon`, line 964) is different: per-batch single-source `[1,1,C+1]`, sign-SGD for 20 steps with step size 0.1, all-reduce-averaged across ranks.

## 6. Hyperparameters (with values from the entry-point configs)

| Field | pile_4L (paper default) | simplestories_2L | Notes |
|---|---|---|---|
| `n_steps` | 400_000 | 400_000 | |
| `batch_size` (global) | 64 | 64 | divided across ranks |
| `seq_len` | 512 | 512 | |
| `main_lr` / `final_frac` | 5e-5 / 0.1 | 3e-4 / 0.1 | cosine, no warmup |
| `faithfulness_warmup_steps` / `_lr` | 400 / 1e-3 | 400 / 1e-3 | components-only, pre-DDP |
| `coeff_faith` | 1e7 | 1e7 | tuned to match other losses' scale |
| `coeff_imp` | 2e-4 | 1e-3 | |
| `coeff_stoch`, `coeff_ppgd` | 0.5, 0.5 | 0.5, 0.5 | |
| `p_start, p_end` | 2.0 → 0.4 | same | linear anneal over `n_steps` |
| `imp_eps, imp_beta` | 1e-12, 0.5 | same | |
| `leaky_alpha` | 0.01 | same | leaky-hard sigmoid slope outside [0,1] |
| `ci_d_model / n_blocks / n_heads / mlp_hidden` | 2048 / 8 / 16 / 8192 | 512 / 4 / 8 / 2048 | scale CI to model size |
| `ppgd_lr` (cosine, warmup 0.025) | 0.01 | 0.01 | beta1=0.5, beta2=0.99 |
| `ppgd_inner_steps` | 2 | 2 | warmup PGD updates per train step |
| `grad_clip_components` | 0.01 | 0.01 | clip on V/U params |
| `eval_freq / slow_eval_freq` | 1000 / 10000 | same | |
| `pgd_eval_step_size, pgd_eval_n_steps` | 0.1, 20 | same | sign-SGD, eval-only |
| `C_per_module` | 24 modules × {512..3584} | 12 modules × {288..1152} | required, model-specific |

## 7. What's directly reusable for a custom 2-layer bilinear transformer

**Copy verbatim:**
- `lower_leaky`, `upper_leaky`, the autograd Function (B).
- `ComponentLinear` and `install_components` (C) — generic over `nn.Linear`. Even a fully-bilinear model has `nn.Linear` somewhere (the up/down projections, embed/unembed).
- The CI transformer (D) — completely independent of the target model. Only needs `[B,S,d_in]` cached inputs, regardless of how the target uses them.
- All loss functions and mask sampling (E) — pure functions of dicts of tensors.
- `PersistentPGD` (F) — generic.
- `cosine_lr` (G), `init_dist`, `SPDModule` (H).
- The main `decompose` training loop (I) up to and including the eval-call site, almost verbatim.

**Needs adaptation:**
- `cfg.C_per_module`: list your model's actual `nn.Linear` paths. For a 2-layer bilinear transformer typical names would be `blocks.0.attn.W_in`, `blocks.0.attn.W_out`, `blocks.0.mlp.W_in`, `blocks.0.mlp.W_out` (or whatever your model uses) — verify with `[name for name, m in model.named_modules() if isinstance(m, nn.Linear)]`.
- Target `model.forward(input_ids) -> logits`: monkey-patch as the entry-point files do if your model returns a tuple.
- Data loader: replace HF `datasets`/tokenizer with whatever your training corpus uses; just yield `[local_B, seq_len]` int64 tensors.

## 8. Gotchas for a bilinear target

- **Bilinear ops aren't `nn.Linear`.** A bilinear MLP `(W_in_a x) ⊙ (W_in_b x)` decomposes through the two projections (each is `nn.Linear`), but the elementwise product is NOT decomposed. The `ComponentLinear` wrapper only intercepts linear layers — the bilinear interaction sits between two wrapped linears, which is exactly what you want, but make sure both `W_in_a` and `W_in_b` are listed in `C_per_module` (similarly two QK pairs in bilinear attention).
- **No biases** in your model: fine — `ComponentLinear` already handles `bias is None` (line 229–232) by registering bias as a buffer of `None`.
- **Channel-scale norm vs RMSNorm/LayerNorm**: irrelevant to PD itself — norms aren't decomposed. The CI transformer uses its own internal `F.rms_norm` of the cached pre-weight inputs; that does NOT have to match your model's norm.
- **Softmax attention assumption**: the CI transformer uses softmax SDPA internally, but that's separate from the target. The target just has to be differentiable and accept `[B,S]` ids. If your bilinear-attention model also has its own attention, that's irrelevant to the wrapper.
- **GELU MLP assumption**: only the CI transformer's internal MLP is GELU; the target model's nonlinearity (or lack thereof, in the bilinear case) is unconstrained.
- **HF/tokenizer assumptions** are confined to the entry-point files (`make_loader`, `load_*_target_model`). Replace those with your own data pipeline yielding int64 `[local_B, seq_len]` tensors.
- **Reconstruction loss is KL on logits**, which assumes the target output is a distribution over a vocab. If you eventually decompose a non-LM bilinear model, swap `kl_logits` for MSE on outputs.
- **`bf16 autocast`** wraps the forward + losses; the total-loss summation is computed outside autocast to keep the coefficient×loss sum in fp32 (line 789). Keep that pattern; bilinear ops can be sensitive to bf16 underflow.
- **Faithfulness coefficient is huge (1e7)** because the loss is normalised to per-element MSE on weights — your weight scale will determine the right coefficient; expect to retune.
- **Importance minimality `world_size` rescaling** (line 435) assumes per-rank batches are independent samples of the same distribution. If you run single-GPU, world_size = 1 and the term simplifies to per-batch.
