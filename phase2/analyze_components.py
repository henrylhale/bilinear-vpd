"""Analyze a trained vpd_v16 component model.

For each (module, component) pair, compute average causal-importance score
on positions where each primitive fires (bigram / skip / induction / filler).
Identify per-primitive selective components.

   PYTHONPATH=. ~/miniconda3/envs/vpd/bin/python -m phase2.analyze_components \
       --target runs/v16_chan_lr3e3 \
       --vpd runs/vpd_v16_B \
       --n_sequences 1024
"""
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from phase1.config import ModelConfig, default_dgp
from phase1.data import (
    ANN_BIGRAM,
    ANN_INDUCTION,
    ANN_NAMES,
    ANN_NONE,
    ANN_SKIP,
    DGP,
)
from phase1.model import BilinearTransformer
from phase2.run_decomposition import build_component_model, load_target_model
from phase2.config import default_vpd_config


@dataclass
class ComponentStats:
    module: str
    component: int
    mean_by_primitive: dict[str, float]  # mean g_c over positions of each primitive

    def selectivity(self) -> tuple[str, float]:
        """Return (best primitive, selectivity-score)."""
        # ratio of best-primitive firing rate to background (filler)
        bg = max(self.mean_by_primitive.get("filler", 0.0), 1e-6)
        best = max(
            (name, val / bg if name != "filler" else 1.0)
            for name, val in self.mean_by_primitive.items()
        )
        return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Phase-1 run dir")
    parser.add_argument("--vpd", required=True, help="Phase-2 run dir with component_model_final.pt + vpd_config.json")
    parser.add_argument("--n_sequences", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    target_run = Path(args.target)
    vpd_run = Path(args.vpd)

    target, model_cfg = load_target_model(target_run)
    device = torch.device(args.device)
    target = target.to(device)

    vpd_cfg = default_vpd_config(target_run_dir=str(target_run), out_dir=str(vpd_run))
    saved_cfg = json.loads((vpd_run / "vpd_config.json").read_text())
    for k, v in saved_cfg.items():
        setattr(vpd_cfg, k, v)

    cm = build_component_model(target, vpd_cfg).to(device)
    sd = torch.load(vpd_run / "component_model_final.pt", map_location=device, weights_only=True)
    cm.load_state_dict(sd)
    cm.eval()

    dgp = DGP(default_dgp(seed=0))
    rng = np.random.default_rng(123)

    print(f"Sampling {args.n_sequences} sequences (seq_len={model_cfg.seq_len})…")
    all_toks, all_anns = [], []
    for i in range(args.n_sequences):
        t, a, _ = dgp.sample_sequence(rng)
        all_toks.append(t)
        all_anns.append(a)
    toks = torch.from_numpy(np.stack(all_toks)).to(device)
    anns = np.stack(all_anns)

    print("Forward pass + computing causal importance…")
    with torch.no_grad():
        bs = 256
        ci_collected: dict[str, list[torch.Tensor]] = defaultdict(list)
        for start in range(0, toks.shape[0], bs):
            end = min(start + bs, toks.shape[0])
            target_out = cm(toks[start:end], cache_type="input")
            ci = cm.calc_causal_importances(target_out.cache, sampling="continuous")
            for mod_name, g in ci.upper_leaky.items():
                ci_collected[mod_name].append(g.cpu())

    # Concat per-module CI tensors: shape (N_seqs, S, C)
    ci_full: dict[str, torch.Tensor] = {
        name: torch.cat(parts, dim=0) for name, parts in ci_collected.items()
    }

    # Annotation masks (N_seqs, S)
    prim_masks = {
        "bigram": anns == ANN_BIGRAM,
        "skip": anns == ANN_SKIP,
        "induction": anns == ANN_INDUCTION,
        "filler": anns == ANN_NONE,
    }
    print(f"\nPositions per primitive across {args.n_sequences} sequences:")
    for name, m in prim_masks.items():
        print(f"  {name:>9s}: {m.sum():>6d}")

    print(f"\nPer-module mean ci_upper across components, broken down by primitive (firing rate proxies):")
    print(f"  {'module':<28s} {'#C':>4s}  {'bigram':>8s} {'skip':>8s} {'induct':>8s} {'filler':>8s}")
    per_module_means: dict[str, dict[str, np.ndarray]] = {}
    for mod_name, ci_tensor in ci_full.items():
        ci_np = ci_tensor.numpy()  # (N_seqs, S, C)
        n_seqs, S, C = ci_np.shape
        means_by_prim = {}
        for prim, mask in prim_masks.items():
            sel = ci_np[mask]  # (n_positions, C)
            if sel.size == 0:
                means_by_prim[prim] = np.zeros(C, dtype=np.float64)
            else:
                means_by_prim[prim] = sel.mean(axis=0)  # per-component mean
        per_module_means[mod_name] = means_by_prim
        agg = {p: means_by_prim[p].mean() for p in ["bigram", "skip", "induction", "filler"]}
        print(f"  {mod_name:<28s} {C:>4d}  {agg['bigram']:>8.3f} {agg['skip']:>8.3f} {agg['induction']:>8.3f} {agg['filler']:>8.3f}")

    # Identify per-component selectivity
    print(f"\nTop-3 most selective components in each module (selectivity = best-primitive-mean / filler-mean):")
    print(f"  {'module':<28s}  {'comp':>4s}  {'best primitive':<10s}  {'mean':>6s}  {'filler-mean':>11s}  {'selectivity':>11s}")
    for mod_name, means_by_prim in per_module_means.items():
        C = len(means_by_prim["filler"])
        sel_list = []
        for c in range(C):
            bg = max(means_by_prim["filler"][c], 1e-6)
            best_prim = max(
                ["bigram", "skip", "induction"],
                key=lambda p: means_by_prim[p][c] / bg,
            )
            best_mean = means_by_prim[best_prim][c]
            sel = best_mean / bg
            sel_list.append((c, best_prim, best_mean, bg, sel))
        sel_list.sort(key=lambda x: -x[4])
        for c, prim, mean, bg, sel in sel_list[:3]:
            print(f"  {mod_name:<28s}  {c:>4d}  {prim:<10s}  {mean:>6.3f}  {bg:>11.4f}  {sel:>11.2f}")

    # Component-count-per-primitive summary
    print("\nPer-primitive 'specialized' components per module (selectivity > 3.0):")
    print(f"  {'module':<28s}  {'bigram':>7s}  {'skip':>7s}  {'induction':>10s}")
    for mod_name, means_by_prim in per_module_means.items():
        C = len(means_by_prim["filler"])
        counts = {"bigram": 0, "skip": 0, "induction": 0}
        for c in range(C):
            bg = max(means_by_prim["filler"][c], 1e-6)
            for prim in counts:
                if means_by_prim[prim][c] / bg > 3.0:
                    counts[prim] += 1
        print(f"  {mod_name:<28s}  {counts['bigram']:>7d}  {counts['skip']:>7d}  {counts['induction']:>10d}")


if __name__ == "__main__":
    main()
