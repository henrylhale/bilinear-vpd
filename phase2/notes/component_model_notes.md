# VPD Framework Reference - applied to a custom BilinearTransformer

Files cited (all under `/home/henry/bilinear-vpd/param_decomp/`):
- `models/component_model.py`
- `models/components.py`
- `losses.py` (dispatcher) + `metrics/{faithfulness,importance_minimality,stochastic_recon,ci_masked_recon,...}_loss.py`
- `run_param_decomp.py`
- `configs.py`
- `utils/module_utils.py` (`expand_module_patterns`, `ModulePathInfo`)
- `utils/component_utils.py` (`calc_stochastic_component_mask_info`)
- `models/batch_and_loss_fns.py` (`recon_loss_kl`, `recon_loss_mse`, `make_run_batch`)
- `experiments/lm/{lm_decomposition.py, ss_llama_simple_mlp-2L.yaml}`

## 1. ComponentModel API

Constructor (`component_model.py:110`):
```
ComponentModel(target_model, run_batch, module_path_info, ci_config, sigmoid_type)
```
- `target_model: nn.Module` — must already be in `eval()` and have `requires_grad_(False)` (asserted, line 121).
- `run_batch: RunBatch` — `(model, batch) -> Tensor`. Built with `make_run_batch(output_extract)`. For a model whose `forward(tokens) -> logits` returns a Tensor directly, use `run_batch_passthrough` (set `output_extract=None`).
- `module_path_info: list[ModulePathInfo]` — each `ModulePathInfo(module_path, C)` says "decompose submodule at this dotted path into C components". Built from a list of `ModulePatternInfoConfig{module_pattern, C}` via `expand_module_patterns(model, ...)`, which resolves fnmatch patterns against `model.named_modules()`.
- `ci_config`: `LayerwiseCiConfig` (one CI fn per layer) or `GlobalCiConfig` (one shared CI fn over all layers).
- `sigmoid_type`: one of `"normal" | "hard" | "leaky_hard" | "upper_leaky_hard" | "swish_hard"`. With `"leaky_hard"`, the model uses a `lower_leaky_hard` sigmoid for masking (CI used in recon) and `upper_leaky_hard` for the importance-minimality penalty.

`from_pretrained(path)` (line 567): one-shot loader for a *trained `ComponentModel` checkpoint*, NOT for raw target weights. It calls `ParamDecompRunInfo.from_path(path)` (loads `final_config.yaml` + `model_*.pt`), then `from_run_info`, which:
1. Resolves `config.pretrained_model_class` and calls its `from_pretrained` (or `from_run_info`) to get the target model;
2. `target_model.eval(); target_model.requires_grad_(False)`;
3. Optionally inserts identity modules (`config.identity_module_info`);
4. Calls `expand_module_patterns` and constructs the wrapper;
5. `torch.load` the state dict, pass through `handle_deprecated_state_dict_keys_`, validate CI compatibility, `load_state_dict`.

Wrapped module types (see `_create_component`, line 190): `nn.Linear`, `nn.Embedding`, HuggingFace `transformers.pytorch_utils.Conv1D` (Radford), and the framework's own `Identity`. Anything else raises `ValueError`. There is no automatic recursion; you must list each leaf yourself via patterns.

Addressing: dotted submodule paths from `model.named_modules()`, exactly as PyTorch produces them. Examples for the bilinear model: `"embed"`, `"unembed"`, `"blocks.0.attn.q1_proj"`, `"blocks.1.mlp.w_n"`. Internally stored in `_components: nn.ModuleDict` with `.` replaced by `-` (purely for state-dict key compatibility).

## 2. Component subclasses

Base `Components` (`components.py:369`) holds `V: (v_dim, C)` and `U: (C, u_dim)` — these are the C_i tensors. Each component i is rank-1: weight delta = `V[:, i:i+1] @ U[i:i+1, :]`. The full reconstructed weight is `V @ U`.

- `LinearComponents` (line 409): for `nn.Linear` and `Conv1D` (and `Identity`).
  - `V: (d_in, C)`, `U: (C, d_out)`.
  - `weight` property returns `V @ U` then transposed to `(d_out, d_in)` to match `nn.Linear`.
  - `bias`: registered as a (frozen) buffer copied from the target — biases are NOT trained.
  - `get_component_acts(x) = x @ V` of shape `(..., C)`. Importance per component is the magnitude of these scalar acts (CI fn maps these to a `[0, 1]` mask).
  - `forward(x, mask, weight_delta_and_mask)`: computes `acts = x @ V`, multiplies by mask if given, then `acts @ U`, optionally adds `weight_delta @ x` weighted by `weight_delta_mask`, then bias.
- `EmbeddingComponents` (line 483): for `nn.Embedding`.
  - `V: (vocab_size, C)`, `U: (C, embedding_dim)`.
  - `get_component_acts(token_ids) = V[token_ids]` shape `(..., C)` (no one-hot).
- `Identity` (line 551): a placeholder `nn.Module` for the identity-insertion mechanism (`identity_insertion.py`) — only needed if you want to decompose places without a real linear layer.

## 3. Forward pass

`ComponentModel(batch, mask_infos=None, cache_type="none")` (line 370):
1. Move batch to device.
2. Fast path: if `mask_infos is None` and `cache_type == "none"`, return `self._run_batch(self.target_model, batch)` — pure forward through the original target model, no hooks.
3. Otherwise, for each module path in either `mask_infos.keys()` or `target_module_paths`, register a forward hook on the target submodule. The hook (`_components_and_cache_hook`):
   - Optionally caches the input `x` (`cache_type="input"`).
   - If `mask_info` is given, calls `self.components[name](x, mask=mask_info.component_mask, weight_delta_and_mask=mask_info.weight_delta_and_mask)` to get `components_out`. The original layer's `output` is replaced by `components_out` everywhere `routing_mask == "all"`, else `torch.where(routing_mask, components_out, output)`.
   - Optionally caches component acts or final output.
4. Hooks are removed after the forward.

Mask sampling does NOT happen inside the forward — it happens outside, in `calc_stochastic_component_mask_info` (`utils/component_utils.py:10`):
```
source = uniform(0,1)  (or randint(0,1) for binomial)
mask  = ci + (1 - ci) * source
```
Causal importances `ci` are produced separately via `model.calc_causal_importances(pre_weight_acts, sampling)` from the cached layer inputs — that's why the typical step is two forward passes: first `cache_type="input"` to grab pre-weight acts, then construct masks, then a second forward with `mask_infos=...`.

## 4. Loss functions (math)

`losses.py` is just a `match` dispatcher. Implementations are in `param_decomp/metrics/*`.

- **FaithfulnessLoss** (`metrics/faithfulness_loss.py:33`): mean squared element of `weight_deltas` aggregated across all decomposed layers.
  `L_faith = (sum_l ||W_l^target - V_l U_l||_F^2) / (sum_l numel(W_l))`.
  Only meaningful when `use_delta_component=False`; otherwise the delta-component absorbs the difference and faithfulness is decorative.
- **ImportanceMinimalityLoss** (`metrics/importance_minimality_loss.py:117`): per-layer
  `L_min = sum_l sum_c [m_{l,c} + beta * m_{l,c} * log2(1 + S_{l,c})]`,
  where `m_{l,c} = mean_{batch,seq}((ci_upper_leaky_{l,c} + eps)^p)` and `S_{l,c}` is the per-component sum (`m * n_examples`). `p` is annealed from `pnorm` toward `p_anneal_final_p` over the window `[p_anneal_start_frac, p_anneal_end_frac]`. Layers are summed (not averaged) — see comment at line 67.
- **UnmaskedReconLoss**: `recon_fn(model(batch, mask_infos=None), target_out) / n_examples`. Sanity check that components leave the target unchanged when no masking is applied.
- **CIMaskedReconLoss / CIMaskedReconSubsetLoss / CIMaskedReconLayerwiseLoss**: deterministic — set `mask = ci.lower_leaky` directly and run a forward pass through components, then `recon_loss(out, target_out)`. The `Subset` variant only routes a `routing_mask`-selected subset of layers to components; `Layerwise` does it for one layer at a time and sums.
- **StochasticReconLoss / StochasticReconSubsetLoss / StochasticReconLayerwiseLoss** (`metrics/stochastic_recon_loss.py:54`): for `n_mask_samples` repetitions, draw `mask = ci + (1-ci)*u`, optionally include weight-delta with random scalar mask, run forward, accumulate `recon_loss`. Average at end. This is the main training driver in the standard setup.
- **PGDReconLoss / Subset / Layerwise**: like Stochastic, but masks are produced by a few PGD steps that *adversarially* maximize the recon loss inside the `[ci, 1]` interval. Knobs: `init` (random|ones|zeroes), `step_size`, `n_steps` (a.k.a. `n_pgd_steps`), `mask_scope`.
- **PersistentPGDReconLoss / Subset**: same idea but the adversarial sources are *learnable parameters* updated once per training step (own optimizer + LR schedule), with a `start_frac` warmup gate.
- **PGDMultiBatchReconLoss / Subset**: PGD across multiple batches with gradient accumulation (used at eval time).
- **StochasticHiddenActsReconLoss** (`metrics/hidden_acts_recon_loss.py`): MSE between hidden activations of the original target and the masked component model at each decomposed layer's *output* — a layerwise reconstruction objective at activation rather than logit level.

The actual scalar reconstruction function is config'd by the experiment driver, not the loss: `recon_loss_kl` (KL of softmaxed logits) for LM; `recon_loss_mse` for TMS/ResidMLP.

## 5. `optimize()` loop (`run_param_decomp.py:117`)

High level:
1. Optionally insert identity modules.
2. `target_model.requires_grad_(False)`; `expand_module_patterns(...)`; build `ComponentModel`; move to device; wrap in DDP if distributed.
3. Optionally tie weights between two component layers (transpose-tied).
4. Collect `component_params = sum_l components[l].parameters()` (i.e. `V`s and `U`s) and `ci_fn_params = ci_fn.parameters()`. **These are the only optimized parameters.** AdamW with `lr=lr_schedule.start_val`, `weight_decay=0`.
5. **Faithfulness warmup** (`run_faithfulness_warmup`, line 65): if `faithfulness_warmup_steps > 0`, run that many AdamW steps minimizing `faithfulness_loss(weight_deltas)` only — initializes `V, U` to a good rank-decomposition of the target weights.
6. Initialize `PersistentPGDState` for any persistent-PGD configs.
7. Main loop for `steps` iterations: update LR per `lr_schedule` (warmup_pct, cosine/linear/constant decay to `final_val_frac * start_val`), zero grads, fetch batch, run target model with `cache_type="input"` to get pre-weight acts and target output, compute CI from acts, call `compute_losses` over all `loss_metric_configs` (each weighted by its `coeff`), backward, optional grad-clip on components and CI fns separately, optimizer step. Persistent-PGD source updates happen alongside.
8. Periodic eval and checkpoint saving.

## 6. Config schema for an LM experiment (the ~10 fields you actually set)

From `Config` in `configs.py:696`, with reference yaml `experiments/lm/ss_llama_simple_mlp-2L.yaml`:

- `pretrained_model_class` + (`pretrained_model_path` or `pretrained_model_name`): how to find your target model.
- `output_extract`: `None` if `forward(input) -> Tensor`, `0` for tuple-returning models, `"logits"` for HF models.
- `module_info: list[{module_pattern, C}]` — fnmatch patterns to decompose, each with its own `C`.
- `ci_config`: `mode: layerwise` with `fn_type: vector_mlp` and `hidden_dims: [...]` is the simplest. Set `fn_type: mlp` for embeddings.
- `loss_metric_configs`: typical set is `FaithfulnessLoss` (huge coeff like 1e7 if `use_delta_component=False`), `ImportanceMinimalityLoss` (`pnorm`, `beta`, `p_anneal_*`), and one of `StochasticReconSubsetLoss` / `StochasticReconLoss` / `PersistentPGDReconLoss` (this last one needs `optimizer`, `scope`, `n_warmup_steps`, `start_frac`, `n_samples`).
- `n_mask_samples`: 1 is fine when you have a strong recon term.
- `use_delta_component: true` strongly recommended — adds a residual that absorbs `W_target - VU`, making faithfulness optional.
- `lr_schedule: {start_val, warmup_pct, final_val_frac, fn_type}`, `steps`, `batch_size`.
- `faithfulness_warmup_steps` (a few hundred is typical when `use_delta_component=false`; set to 0 otherwise).
- `task_config`: for an LM use `LMTaskConfig{task_name: lm, max_seq_len, dataset_name, column_name, ...}`.
- `tokenizer_name`.
- Logging: `train_log_freq`, `eval_freq`, `slow_eval_freq`, `n_eval_steps`, `eval_batch_size`, `save_freq`.

PGD-specific knobs: `init`, `step_size`, `n_steps` (PGD inner steps per outer step), `mask_scope`.

## 7. Assumptions about the target model

Concretely:
- Target is an `nn.Module` whose `forward` returns a Tensor (or tuple/object navigable by `output_extract`).
- All parameters must satisfy `requires_grad == False` at construction time (`component_model.py:122` — assertion).
- Submodules to decompose must be one of: `nn.Linear`, `nn.Embedding`, `transformers.pytorch_utils.Conv1D`, framework `Identity`. Each must be addressable by a single dotted name and produce a single-tensor output (`_components_and_cache_hook` asserts `len(args)==1`, no kwargs, `isinstance(output, Tensor)`).
- No requirement on layernorm / RMSNorm / softmax / biases. The framework only ever inserts hooks on the listed leaf modules and does not introspect surrounding logic. Norms/scales/activations stay as-is.

For your `BilinearTransformer` (`/home/henry/bilinear-vpd/phase1/model.py`):
- `embed: nn.Embedding`, `unembed: nn.Linear`, and every Q1/Q2/K1/K2/V/O proj plus `w_m`, `w_n`, `w_proj` are `nn.Linear(bias=False)`. All are wrappable.
- `RMSNorm`, `LearnableScalar`, `LearnableChannelScale` are pure non-decomposed `nn.Module`s — they just sit in the residual stream and the framework leaves them alone. Their `.scale` parameters MUST also be frozen (`requires_grad=False`) before passing to `ComponentModel`. If you want them to remain trainable for some reason, you'd have to whitelist them — by default the constructor asserts every param is frozen.
- The two QK pairs and bilinear MLP arms (`w_m`, `w_n`, with `out = w_proj(w_m(x) * w_n(x))`) are no problem: each `nn.Linear` is independently decomposed.
- No biases, no softmax-attention quirks, no layernorm — none of these block VPD.
- The two unusual things to watch out for: (a) the `register_buffer`s `rope_cos`, `rope_sin`, `causal_mask` won't move automatically when you call `.to(device)` on the wrapper — check that `next(model.parameters()).device` matches buffers; (b) your `forward(tokens) -> logits` returns a Tensor directly, so `output_extract=None` and `run_batch=run_batch_passthrough`.

## 8. Smoke-test snippet

```python
import torch
from param_decomp.configs import (
    LayerwiseCiConfig, ModulePatternInfoConfig, ImportanceMinimalityLossConfig,
    StochasticReconLossConfig, FaithfulnessLossConfig,
)
from param_decomp.models.batch_and_loss_fns import run_batch_passthrough, recon_loss_kl
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import make_mask_infos
from param_decomp.utils.component_utils import calc_stochastic_component_mask_info
from param_decomp.utils.module_utils import expand_module_patterns
from param_decomp.routing import AllLayersRouter
from param_decomp.losses import compute_losses

from phase1.model import BilinearTransformer, ModelConfig  # your code

device = "cuda"
ckpt = torch.load("/home/henry/bilinear-vpd/runs/v12_chan_long/model_final.pt",
                  map_location=device, weights_only=True)

cfg = ModelConfig(...)  # TODO: load from config.json next to the .pt
target = BilinearTransformer(cfg).to(device)
target.load_state_dict(ckpt)
target.eval()
target.requires_grad_(False)

patterns = [
    ModulePatternInfoConfig(module_pattern="embed",                    C=64),
    ModulePatternInfoConfig(module_pattern="unembed",                  C=64),
    ModulePatternInfoConfig(module_pattern="blocks.*.attn.q1_proj",    C=32),
    ModulePatternInfoConfig(module_pattern="blocks.*.attn.q2_proj",    C=32),
    ModulePatternInfoConfig(module_pattern="blocks.*.attn.k1_proj",    C=32),
    ModulePatternInfoConfig(module_pattern="blocks.*.attn.k2_proj",    C=32),
    ModulePatternInfoConfig(module_pattern="blocks.*.attn.v_proj",     C=32),
    ModulePatternInfoConfig(module_pattern="blocks.*.attn.o_proj",     C=32),
    ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_m",         C=64),
    ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_n",         C=64),
    ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_proj",      C=64),
]
module_path_info = expand_module_patterns(target, patterns)

ci_config = LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[16])

cm = ComponentModel(
    target_model=target,
    run_batch=run_batch_passthrough,        # forward(tokens) -> logits Tensor
    module_path_info=module_path_info,
    ci_config=ci_config,
    sigmoid_type="leaky_hard",
).to(device)

tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), device=device)

# Pass 1: cache pre-weight acts + raw target output.
target_out = cm(tokens, cache_type="input")     # OutputWithCache
ci = cm.calc_causal_importances(target_out.cache, sampling="continuous")

# Pass 2 (smoke): one stochastic forward.
weight_deltas = cm.calc_weight_deltas()
mask_infos = calc_stochastic_component_mask_info(
    causal_importances=ci.lower_leaky, component_mask_sampling="continuous",
    weight_deltas=weight_deltas, router=AllLayersRouter(),
)
out_masked = cm(tokens, mask_infos=mask_infos)

# One loss: stochastic recon + importance minimality.
losses = compute_losses(
    loss_metric_configs=[
        StochasticReconLossConfig(coeff=1.0),
        ImportanceMinimalityLossConfig(coeff=1e-3, pnorm=2.0, beta=0.5),
        FaithfulnessLossConfig(coeff=1e7),
    ],
    model=cm, batch=tokens, ci=ci, target_out=target_out.output,
    weight_deltas=weight_deltas, current_frac_of_training=0.0,
    sampling="continuous", use_delta_component=True, n_mask_samples=1,
    ppgd_states={}, reconstruction_loss=recon_loss_kl,
)
total = sum(c.coeff * v for c, v in losses.items())
print({type(k).__name__: v.item() for k, v in losses.items()}, total.item())
```

## Key takeaways / integration risks for the bilinear model

1. The framework is happy with arbitrary `nn.Module` so long as decomposable layers are `nn.Linear` / `nn.Embedding` / `Conv1D` / `Identity`. RMSNorm, learnable scalars, channel-scale norms, lack of softmax, lack of biases, and the bilinear `(w_m·w_n)` motif all pass through the wrapper untouched.
2. Freeze every parameter on the target model (including any `LearnableScalar` / `LearnableChannelScale.scale`) before constructing — the constructor asserts this.
3. Use `output_extract=None` and `run_batch_passthrough` since `forward(tokens) -> Tensor`.
4. Use `recon_loss_kl` for an LM logits objective (or MSE on hidden acts if you prefer).
5. `from_pretrained` is for *trained ComponentModel checkpoints*, not target models; for a fresh decomposition you instantiate `ComponentModel(target, ...)` yourself, exactly as `optimize` does.
6. RoPE buffers are non-persistent — they'll be on the wrapper after `.to(device)` because they're registered on the target sub-module, but if you reload weights manually be careful not to wipe them.
7. `pre_identity` suffix is only present if you opt into `identity_module_info`; otherwise you address the existing leaves directly.
