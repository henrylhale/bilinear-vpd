"""Phase-2 smoke test: load a Phase-1 checkpoint, wrap it with `ComponentModel`,
run one forward pass with stochastic masks + faithfulness/recon/minimality
losses, verify nothing explodes.

Not a full decomposition run — just enough to confirm the BilinearTransformer
plumbs through `param_decomp` cleanly. Run from repo root with:

    PYTHONPATH=. ~/miniconda3/envs/vpd/bin/python -m phase2.smoke_test
"""
import json
from dataclasses import asdict
from pathlib import Path

import torch

from param_decomp.configs import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    StochasticReconLossConfig,
)
from param_decomp.losses import compute_losses
from param_decomp.models.batch_and_loss_fns import recon_loss_kl, run_batch_passthrough
from param_decomp.models.component_model import ComponentModel
from param_decomp.routing import AllLayersRouter
from param_decomp.utils.component_utils import calc_stochastic_component_mask_info
from param_decomp.utils.module_utils import expand_module_patterns

from phase1.config import ModelConfig
from phase1.model import BilinearTransformer

CHECKPOINT = Path("runs/v12_chan_long/model_final.pt")
CONFIG_JSON = Path("runs/v12_chan_long/config.json")


def load_target_model() -> tuple[BilinearTransformer, ModelConfig]:
    raw = json.loads(CONFIG_JSON.read_text())
    cfg = ModelConfig(**raw["model"])
    model = BilinearTransformer(cfg)
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    model.requires_grad_(False)
    return model, cfg


def main() -> None:
    target, model_cfg = load_target_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = target.to(device)
    print(f"target loaded ({sum(p.numel() for p in target.parameters()):,} frozen params) on {device}")

    patterns = [
        ModulePatternInfoConfig(module_pattern="embed", C=64),
        ModulePatternInfoConfig(module_pattern="unembed", C=64),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.q1_proj", C=32),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.q2_proj", C=32),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.k1_proj", C=32),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.k2_proj", C=32),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.v_proj", C=32),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.o_proj", C=32),
        ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_m", C=64),
        ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_n", C=64),
        ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_proj", C=64),
    ]
    module_path_info = expand_module_patterns(target, patterns)
    print(f"resolved {len(module_path_info)} target modules to decompose:")
    for info in module_path_info:
        print(f"  - {info.module_path}  C={info.C}")

    ci_config = LayerwiseCiConfig(fn_type="mlp", hidden_dims=[16])
    cm = ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        module_path_info=module_path_info,
        ci_config=ci_config,
        sigmoid_type="leaky_hard",
    ).to(device)
    n_trainable = sum(p.numel() for p in cm.parameters() if p.requires_grad)
    print(f"ComponentModel built; {n_trainable:,} trainable params (V/U + CI fns)")

    tokens = torch.randint(0, model_cfg.vocab_size, (4, model_cfg.seq_len), device=device)

    target_out = cm(tokens, cache_type="input")
    print(f"pass 1 (cache): output shape {tuple(target_out.output.shape)}, "
          f"cached {len(target_out.cache)} layer inputs")

    ci = cm.calc_causal_importances(target_out.cache, sampling="continuous")
    weight_deltas = cm.calc_weight_deltas()
    mask_infos = calc_stochastic_component_mask_info(
        causal_importances=ci.lower_leaky,
        component_mask_sampling="continuous",
        weight_deltas=weight_deltas,
        router=AllLayersRouter(),
    )

    out_masked = cm(tokens, mask_infos=mask_infos)
    print(f"pass 2 (masked): output shape {tuple(out_masked.shape)}, "
          f"finite={bool(torch.isfinite(out_masked).all())}")

    loss_configs = [
        FaithfulnessLossConfig(coeff=1e7),
        StochasticReconLossConfig(coeff=1.0),
        ImportanceMinimalityLossConfig(coeff=1e-3, pnorm=2.0, beta=0.5),
    ]
    losses = compute_losses(
        loss_metric_configs=loss_configs,
        model=cm,
        batch=tokens,
        ci=ci,
        target_out=target_out.output,
        weight_deltas=weight_deltas,
        current_frac_of_training=0.0,
        sampling="continuous",
        use_delta_component=True,
        n_mask_samples=1,
        ppgd_states={},
        reconstruction_loss=recon_loss_kl,
    )
    print("\nlosses:")
    for k, v in losses.items():
        print(f"  {type(k).__name__:<32s} = {v.item():.4f}")

    total = sum(c.coeff * v for c, v in losses.items())
    total.backward()
    grad_tensors = [p for p in cm.parameters() if p.requires_grad and p.grad is not None]
    grad_with_signal = sum(p.grad.abs().sum().item() > 0 for p in grad_tensors)
    print(f"\ntotal weighted loss = {total.item():.4f}")
    print(f"backward succeeded; {grad_with_signal}/{len(grad_tensors)} parameter tensors got nonzero grad")


if __name__ == "__main__":
    main()
