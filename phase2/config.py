from dataclasses import dataclass, field


@dataclass
class VPDConfig:
    """Hyperparameters for vanilla VPD (no PPGD) on a Phase-1 checkpoint."""

    target_run_dir: str
    out_dir: str

    c_attn_proj: int
    c_mlp: int
    c_embed: int
    ci_hidden: int

    lr: float
    weight_decay: float
    betas: tuple[float, float]
    grad_clip_uv: float
    grad_clip_ci: float

    n_steps: int
    warmup_steps: int
    batch_size: int

    faith_warmup_steps: int
    faith_warmup_lr: float

    coeff_faith: float
    coeff_stoch_recon: float
    coeff_imp: float
    pnorm_start: float
    pnorm_end: float
    pnorm_anneal_start_frac: float
    pnorm_anneal_end_frac: float
    beta_imp: float

    eval_every: int
    eval_n_batches: int
    checkpoint_steps: tuple[int, ...]

    num_data_workers: int
    seed: int


def default_vpd_config(target_run_dir: str, out_dir: str) -> VPDConfig:
    return VPDConfig(
        target_run_dir=target_run_dir,
        out_dir=out_dir,
        c_attn_proj=32,
        c_mlp=64,
        c_embed=64,
        ci_hidden=16,
        lr=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        grad_clip_uv=0.01,
        grad_clip_ci=0.1,
        n_steps=20000,
        warmup_steps=200,
        batch_size=256,
        faith_warmup_steps=400,
        faith_warmup_lr=1e-3,
        coeff_faith=1e7,
        coeff_stoch_recon=0.5,
        coeff_imp=1e-3,
        pnorm_start=2.0,
        pnorm_end=0.4,
        pnorm_anneal_start_frac=0.0,
        pnorm_anneal_end_frac=1.0,
        beta_imp=0.5,
        eval_every=500,
        eval_n_batches=4,
        checkpoint_steps=(500, 1000, 2000, 5000, 10000, 20000),
        num_data_workers=0,
        seed=0,
    )
