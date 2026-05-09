import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phase1.config import RunConfig
from phase1.data import ANN_BIGRAM, ANN_INDUCTION, ANN_SKIP, DGP, make_train_loader
from phase1.eval import evaluate, format_report
from phase1.model import BilinearTransformer


def cosine_lr(step: int, base_lr: float, warmup: int, total: int, min_frac: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return base_lr * min_frac
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_run_config(cfg: RunConfig, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def train(run_cfg: RunConfig) -> None:
    torch.manual_seed(run_cfg.train.seed)
    torch.set_num_threads(1)

    out_dir = Path(run_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_cfg, out_dir / "config.json")

    dgp = DGP(run_cfg.dgp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BilinearTransformer(run_cfg.model).to(device)
    print(f"device: {device}")
    print(f"model params: {model.num_params():,}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=run_cfg.train.lr,
        betas=run_cfg.train.betas,
        weight_decay=run_cfg.train.weight_decay,
    )

    eval_rng = np.random.default_rng(run_cfg.train.seed + 10_000)

    train_loader = make_train_loader(
        run_cfg.dgp,
        batch_size=run_cfg.train.batch_size,
        base_seed=run_cfg.train.seed,
        num_workers=run_cfg.train.num_data_workers,
    )
    train_iter = iter(train_loader)

    log_path = out_dir / "log.jsonl"
    log_f = open(log_path, "w")

    n_steps = run_cfg.train.n_steps
    warmup = run_cfg.train.warmup_steps
    V = run_cfg.model.vocab_size

    start = time.time()
    last_train_loss = float("nan")

    for step in range(n_steps + 1):
        lr = cosine_lr(step, run_cfg.train.lr, warmup, n_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        if step < n_steps:
            model.train()
            toks, _, _ = next(train_iter)
            toks = toks.to(device)
            logits = model(toks)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_train_loss = float(loss.item())
            if step % 100 == 0:
                elapsed = time.time() - start
                print(
                    f"step {step:6d}  lr={lr:.2e}  train_loss={last_train_loss:.4f}  "
                    f"elapsed={elapsed:.0f}s"
                )

        do_eval = (step % run_cfg.train.eval_every == 0) or (step == n_steps)
        if do_eval:
            report = evaluate(
                model,
                dgp,
                eval_rng,
                n_batches=run_cfg.train.eval_n_batches,
                batch_size=run_cfg.train.batch_size,
                device=device,
            )
            print(f"--- eval @ step {step} ---")
            print(format_report(report))
            log_f.write(
                json.dumps(
                    {
                        "step": step,
                        "lr": lr,
                        "train_loss": last_train_loss,
                        "eval_overall_loss": report.overall_loss,
                        "kl_bigram": report.per_primitive[ANN_BIGRAM].mean_kl,
                        "kl_skip": report.per_primitive[ANN_SKIP].mean_kl,
                        "kl_induction": report.per_primitive[ANN_INDUCTION].mean_kl,
                        "h_bigram": report.per_primitive[ANN_BIGRAM].mean_true_entropy,
                        "h_skip": report.per_primitive[ANN_SKIP].mean_true_entropy,
                        "h_induction": report.per_primitive[ANN_INDUCTION].mean_true_entropy,
                        "top1_bigram": report.per_primitive[ANN_BIGRAM].top1_accuracy,
                        "top1_skip": report.per_primitive[ANN_SKIP].top1_accuracy,
                        "top1_induction": report.per_primitive[ANN_INDUCTION].top1_accuracy,
                        "n_bigram": report.per_primitive[ANN_BIGRAM].n_positions,
                        "n_skip": report.per_primitive[ANN_SKIP].n_positions,
                        "n_induction": report.per_primitive[ANN_INDUCTION].n_positions,
                    }
                )
                + "\n"
            )
            log_f.flush()

        if step in run_cfg.train.checkpoint_steps:
            ckpt = out_dir / f"model_step_{step}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  saved checkpoint {ckpt}")

    log_f.close()
    final = out_dir / "model_final.pt"
    torch.save(model.state_dict(), final)
    print(f"saved final {final}")
