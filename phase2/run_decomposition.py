"""Vanilla VPD (no PPGD) on a Phase-1 checkpoint.

   PYTHONPATH=. ~/miniconda3/envs/vpd/bin/python -m phase2.run_decomposition \
       --target runs/v16_chan_lr3e3 --out runs/vpd_v16 --n_steps 20000
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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

from phase1.config import ModelConfig, default_dgp
from phase1.data import DGP, make_train_loader
from phase1.model import BilinearTransformer
from phase2.config import VPDConfig, default_vpd_config


def cosine_lr(step: int, base_lr: float, warmup: int, total: int, min_frac: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return base_lr * min_frac
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def load_target_model(run_dir: Path) -> tuple[BilinearTransformer, ModelConfig]:
    raw = json.loads((run_dir / "config.json").read_text())
    mcfg = ModelConfig(**raw["model"])
    model = BilinearTransformer(mcfg)
    sd = torch.load(run_dir / "model_final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    model.requires_grad_(False)
    return model, mcfg


def build_component_model(target: BilinearTransformer, cfg: VPDConfig) -> ComponentModel:
    patterns = [
        ModulePatternInfoConfig(module_pattern="embed", C=cfg.c_embed),
        ModulePatternInfoConfig(module_pattern="unembed", C=cfg.c_embed),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.q1_proj", C=cfg.c_attn_proj),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.q2_proj", C=cfg.c_attn_proj),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.k1_proj", C=cfg.c_attn_proj),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.k2_proj", C=cfg.c_attn_proj),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.v_proj", C=cfg.c_attn_proj),
        ModulePatternInfoConfig(module_pattern="blocks.*.attn.o_proj", C=cfg.c_attn_proj),
        ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_m", C=cfg.c_mlp),
        ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_n", C=cfg.c_mlp),
        ModulePatternInfoConfig(module_pattern="blocks.*.mlp.w_proj", C=cfg.c_mlp),
    ]
    module_path_info = expand_module_patterns(target, patterns)
    ci_config = LayerwiseCiConfig(fn_type="mlp", hidden_dims=[cfg.ci_hidden])
    return ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        module_path_info=module_path_info,
        ci_config=ci_config,
        sigmoid_type="leaky_hard",
    )


def partition_params(cm: ComponentModel) -> tuple[list, list]:
    """Split trainable params into (V/U component params, CI fn params)."""
    uv, ci = [], []
    for name, p in cm.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("_components."):
            uv.append(p)
        elif name.startswith("ci_fn."):
            ci.append(p)
        else:
            raise RuntimeError(f"unclassified trainable param: {name}")
    return uv, ci


def evaluate(cm: ComponentModel, dgp: DGP, rng: np.random.Generator, batch_size: int, n_batches: int, device: torch.device) -> dict:
    """Quick eval: KL between target output and ci-masked output (no PGD)."""
    cm.eval()
    total_kl = 0.0
    total_n = 0
    total_ci_l0 = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            tokens, _, _ = dgp.sample_batch(rng, batch_size)
            tokens = tokens.to(device)
            target_out = cm(tokens, cache_type="input")
            ci = cm.calc_causal_importances(target_out.cache, sampling="continuous")
            weight_deltas = cm.calc_weight_deltas()
            mask_infos = calc_stochastic_component_mask_info(
                causal_importances=ci.lower_leaky,
                component_mask_sampling="continuous",
                weight_deltas=weight_deltas,
                router=AllLayersRouter(),
            )
            masked = cm(tokens, mask_infos=mask_infos)
            kl = F.kl_div(
                F.log_softmax(masked, dim=-1),
                F.log_softmax(target_out.output, dim=-1),
                log_target=True, reduction="sum",
            )
            total_kl += float(kl.item())
            total_n += int(tokens.numel())
            for layer_ci in ci.upper_leaky.values():
                total_ci_l0 += float((layer_ci > 0.5).float().mean().item())
    cm.train()
    return {
        "eval_recon_kl": total_kl / max(1, total_n),
        "eval_ci_l0_frac": total_ci_l0 / (n_batches * len(cm._components)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Phase-1 run dir with model_final.pt + config.json")
    parser.add_argument("--out", required=True, help="Output dir")
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--coeff_imp", type=float, default=None)
    parser.add_argument("--num_data_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = default_vpd_config(target_run_dir=args.target, out_dir=args.out)
    cfg.seed = args.seed
    if args.n_steps is not None:
        cfg.n_steps = args.n_steps
        cfg.checkpoint_steps = tuple(s for s in cfg.checkpoint_steps if s <= cfg.n_steps) + (cfg.n_steps,)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.coeff_imp is not None:
        cfg.coeff_imp = args.coeff_imp
    if args.num_data_workers is not None:
        cfg.num_data_workers = args.num_data_workers

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vpd_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    target_run = Path(cfg.target_run_dir)
    target, model_cfg = load_target_model(target_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = target.to(device)
    print(f"target loaded ({sum(p.numel() for p in target.parameters()):,} frozen params) on {device}")

    cm = build_component_model(target, cfg).to(device)
    uv_params, ci_params = partition_params(cm)
    print(f"trainable: {sum(p.numel() for p in uv_params):,} UV + {sum(p.numel() for p in ci_params):,} CI")

    opt = torch.optim.AdamW(
        [
            {"params": uv_params, "lr": cfg.lr, "name": "uv"},
            {"params": ci_params, "lr": cfg.lr, "name": "ci"},
        ],
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    dgp = DGP(default_dgp(seed=cfg.seed))
    train_loader = make_train_loader(default_dgp(seed=cfg.seed), batch_size=cfg.batch_size, base_seed=cfg.seed, num_workers=cfg.num_data_workers)
    train_iter = iter(train_loader)
    eval_rng = np.random.default_rng(cfg.seed + 10_000)

    log_f = open(out_dir / "log.jsonl", "w")
    print(f"\n=== Phase A: faithfulness warmup ({cfg.faith_warmup_steps} steps @ lr={cfg.faith_warmup_lr}) ===")
    for g in opt.param_groups:
        g["lr"] = cfg.faith_warmup_lr
    cm.train()
    for step in range(cfg.faith_warmup_steps):
        weight_deltas = cm.calc_weight_deltas()
        loss_faith_raw = sum(((wd ** 2).sum()) for wd in weight_deltas.values()) / sum(wd.numel() for wd in weight_deltas.values())
        loss = cfg.coeff_faith * loss_faith_raw
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(uv_params, cfg.grad_clip_uv)
        opt.step()
        if step % 50 == 0:
            print(f"  warmup step {step:4d}  faith_raw={loss_faith_raw.item():.6e}  weighted={loss.item():.2e}")

    print(f"\n=== Phase B: main VPD loop ({cfg.n_steps} steps) ===")
    start = time.time()
    last_train = float("nan")
    for step in range(cfg.n_steps + 1):
        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, cfg.n_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        if step < cfg.n_steps:
            cm.train()
            toks, _, _ = next(train_iter)
            toks = toks.to(device)
            target_out = cm(toks, cache_type="input")
            ci = cm.calc_causal_importances(target_out.cache, sampling="continuous")
            weight_deltas = cm.calc_weight_deltas()
            mask_infos = calc_stochastic_component_mask_info(
                causal_importances=ci.lower_leaky,
                component_mask_sampling="continuous",
                weight_deltas=weight_deltas,
                router=AllLayersRouter(),
            )
            current_frac = step / max(1, cfg.n_steps)
            losses = compute_losses(
                loss_metric_configs=[
                    FaithfulnessLossConfig(coeff=cfg.coeff_faith),
                    StochasticReconLossConfig(coeff=cfg.coeff_stoch_recon),
                    ImportanceMinimalityLossConfig(
                        coeff=cfg.coeff_imp,
                        pnorm=cfg.pnorm_start,
                        beta=cfg.beta_imp,
                        p_anneal_final_p=cfg.pnorm_end,
                        p_anneal_start_frac=cfg.pnorm_anneal_start_frac,
                        p_anneal_end_frac=cfg.pnorm_anneal_end_frac,
                    ),
                ],
                model=cm, batch=toks, ci=ci, target_out=target_out.output,
                weight_deltas=weight_deltas,
                current_frac_of_training=current_frac,
                sampling="continuous", use_delta_component=True, n_mask_samples=1,
                ppgd_states={}, reconstruction_loss=recon_loss_kl,
            )
            total = sum(c.coeff * v for c, v in losses.items())
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(uv_params, cfg.grad_clip_uv)
            torch.nn.utils.clip_grad_norm_(ci_params, cfg.grad_clip_ci)
            opt.step()
            last_train = float(total.item())
            if step % 100 == 0:
                el = time.time() - start
                parts = {type(c).__name__.replace("LossConfig", ""): f"{v.item():.3e}" for c, v in losses.items()}
                print(f"step {step:6d}  lr={lr:.2e}  total={last_train:.3e}  " + " ".join(f"{k}={v}" for k, v in parts.items()) + f"  ({el:.0f}s)")

        do_eval = (step % cfg.eval_every == 0) or (step == cfg.n_steps)
        if do_eval:
            metrics = evaluate(cm, dgp, eval_rng, cfg.batch_size, cfg.eval_n_batches, device)
            print(f"--- eval @ step {step}: recon_kl={metrics['eval_recon_kl']:.4f}  ci_l0_frac={metrics['eval_ci_l0_frac']:.3f} ---")
            log_f.write(json.dumps({"step": step, "lr": lr, "train_total": last_train, **metrics}) + "\n")
            log_f.flush()

        if step in cfg.checkpoint_steps:
            ckpt = out_dir / f"component_model_step_{step}.pt"
            torch.save(cm.state_dict(), ckpt)
            print(f"  saved {ckpt}")

    final = out_dir / "component_model_final.pt"
    torch.save(cm.state_dict(), final)
    log_f.close()
    print(f"\nDONE. saved {final}")


if __name__ == "__main__":
    main()
