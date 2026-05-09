from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from phase1.data import (
    ANN_BIGRAM,
    ANN_INDUCTION,
    ANN_NAMES,
    ANN_SKIP,
    DGP,
)
from phase1.model import BilinearTransformer


@dataclass
class PerPrimitiveMetrics:
    n_positions: int
    mean_kl: float  # KL(p_true || p_model), nats
    mean_ce: float  # cross-entropy of model under p_true, nats
    mean_true_entropy: float  # H(p_true), nats
    top1_accuracy: float


@dataclass
class EvalReport:
    overall_loss: float  # mean CE over all next-token predictions
    per_primitive: dict[int, PerPrimitiveMetrics]


def _eval_batch(
    model: BilinearTransformer,
    dgp: DGP,
    toks_t: torch.Tensor,
    anns_t: torch.Tensor,
    inds_t: torch.Tensor,
    device: torch.device,
    accum: dict,
) -> tuple[float, int]:
    toks_dev = toks_t.to(device)
    logits = model(toks_dev)
    V = dgp.vocab_size
    log_probs = F.log_softmax(logits, dim=-1)
    targets = toks_dev[:, 1:].reshape(-1)
    preds = logits[:, :-1].reshape(-1, V)
    loss_sum = float(F.cross_entropy(preds, targets, reduction="sum").item())
    n_targets = int(targets.numel())

    anns_np = anns_t.numpy()
    inds_np = inds_t.numpy()
    toks_np = toks_t.numpy()
    log_probs_np = log_probs.detach().cpu().numpy()

    for a in (ANN_BIGRAM, ANN_SKIP, ANN_INDUCTION):
        bs, ts = np.nonzero(anns_np == a)
        for b, t in zip(bs.tolist(), ts.tolist()):
            p_true = dgp.true_distribution(toks_np[b], anns_np[b], inds_np[b], t)
            log_q = log_probs_np[b, t - 1]  # prediction for token at t
            mask = p_true > 0
            kl = float(np.sum(p_true[mask] * (np.log(p_true[mask]) - log_q[mask])))
            ce = float(-np.sum(p_true * log_q))
            h_true = float(-np.sum(p_true[mask] * np.log(p_true[mask])))
            accum["kl"][a] += kl
            accum["ce"][a] += ce
            accum["h"][a] += h_true
            accum["n"][a] += 1
            if int(np.argmax(log_q)) == int(np.argmax(p_true)):
                accum["top1"][a] += 1
    return loss_sum, n_targets


def evaluate(
    model: BilinearTransformer,
    dgp: DGP,
    rng: np.random.Generator,
    n_batches: int,
    batch_size: int,
    device: torch.device,
) -> EvalReport:
    model.eval()
    accum = {
        "kl": {a: 0.0 for a in (ANN_BIGRAM, ANN_SKIP, ANN_INDUCTION)},
        "ce": {a: 0.0 for a in (ANN_BIGRAM, ANN_SKIP, ANN_INDUCTION)},
        "h": {a: 0.0 for a in (ANN_BIGRAM, ANN_SKIP, ANN_INDUCTION)},
        "n": {a: 0 for a in (ANN_BIGRAM, ANN_SKIP, ANN_INDUCTION)},
        "top1": {a: 0 for a in (ANN_BIGRAM, ANN_SKIP, ANN_INDUCTION)},
    }
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for _ in range(n_batches):
            toks_t, anns_t, inds_t = dgp.sample_batch(rng, batch_size)
            ls, n = _eval_batch(model, dgp, toks_t, anns_t, inds_t, device, accum)
            total_loss += ls
            total_n += n
    per: dict[int, PerPrimitiveMetrics] = {}
    for a in (ANN_BIGRAM, ANN_SKIP, ANN_INDUCTION):
        n = accum["n"][a]
        if n == 0:
            per[a] = PerPrimitiveMetrics(0, 0.0, 0.0, 0.0, 0.0)
        else:
            per[a] = PerPrimitiveMetrics(
                n_positions=n,
                mean_kl=accum["kl"][a] / n,
                mean_ce=accum["ce"][a] / n,
                mean_true_entropy=accum["h"][a] / n,
                top1_accuracy=accum["top1"][a] / n,
            )
    return EvalReport(overall_loss=total_loss / max(1, total_n), per_primitive=per)


def format_report(report: EvalReport) -> str:
    lines = [f"  overall_loss: {report.overall_loss:.4f}"]
    for a, m in report.per_primitive.items():
        name = ANN_NAMES[a]
        lines.append(
            f"  {name:10s} n={m.n_positions:5d}  kl={m.mean_kl:.4f}  "
            f"ce={m.mean_ce:.4f}  H_true={m.mean_true_entropy:.4f}  top1={m.top1_accuracy:.3f}"
        )
    return "\n".join(lines)
