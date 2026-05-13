"""For each SUBJ token, find which components fire most when that token is the bigram trigger.

A clean VPD decomposition should produce per-SUBJ component clusters: distinct
components for SUBJ_0, SUBJ_1, ..., each handling that subject's
verb-conditional distribution.

   PYTHONPATH=. ~/miniconda3/envs/vpd/bin/python -m phase2.analyze_per_subj \
       --target runs/v16_chan_lr3e3 --vpd runs/vpd_v16_B --n_sequences 1024
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from phase1.config import ModelConfig, default_dgp
from phase1.data import ANN_BIGRAM, DGP
from phase1.model import BilinearTransformer
from phase2.run_decomposition import build_component_model, load_target_model
from phase2.config import default_vpd_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--vpd", required=True)
    parser.add_argument("--n_sequences", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=[
            "blocks.0.attn.q1_proj",
            "blocks.0.attn.q2_proj",
            "blocks.0.attn.k1_proj",
            "blocks.0.attn.k2_proj",
            "blocks.1.mlp.w_m",
            "blocks.1.mlp.w_n",
            "unembed",
        ],
        help="modules to dump per-SUBJ component firing tables for",
    )
    args = parser.parse_args()

    target_run = Path(args.target)
    vpd_run = Path(args.vpd)
    target, model_cfg = load_target_model(target_run)
    device = torch.device(args.device)
    target = target.to(device)

    vpd_cfg = default_vpd_config(target_run_dir=str(target_run), out_dir=str(vpd_run))
    saved = json.loads((vpd_run / "vpd_config.json").read_text())
    for k, v in saved.items():
        setattr(vpd_cfg, k, v)

    cm = build_component_model(target, vpd_cfg).to(device)
    sd = torch.load(vpd_run / "component_model_final.pt", map_location=device, weights_only=True)
    cm.load_state_dict(sd)
    cm.eval()

    dgp = DGP(default_dgp(seed=0))
    rng = np.random.default_rng(31)
    print(f"Sampling {args.n_sequences} sequences…")
    seqs, anns = [], []
    for _ in range(args.n_sequences):
        t, a, _ = dgp.sample_sequence(rng)
        seqs.append(t)
        anns.append(a)
    toks_all = torch.from_numpy(np.stack(seqs))
    anns_all = np.stack(anns)

    print("Forward pass + CI computation…")
    ci_collected: dict[str, list[torch.Tensor]] = defaultdict(list)
    bs = 256
    with torch.no_grad():
        for s in range(0, toks_all.shape[0], bs):
            e = min(s + bs, toks_all.shape[0])
            out = cm(toks_all[s:e].to(device), cache_type="input")
            ci = cm.calc_causal_importances(out.cache, sampling="continuous")
            for mod, g in ci.upper_leaky.items():
                if mod in args.modules:
                    ci_collected[mod].append(g.cpu())

    # Concatenate
    ci_full = {m: torch.cat(parts, dim=0).numpy() for m, parts in ci_collected.items()}

    seqs_arr = np.stack(seqs)
    bigram_mask = anns_all == ANN_BIGRAM
    subj_start = dgp.vocab.subj_start
    n_subj = dgp.vocab.sizes.n_subj
    print(f"\nBigram-firing positions per SUBJ token (across {args.n_sequences} sequences):")
    per_subj_counts = {}
    for s in range(subj_start, subj_start + n_subj):
        prev_is_s = np.zeros_like(anns_all, dtype=bool)
        prev_is_s[:, 1:] = seqs_arr[:, :-1] == s
        cnt = (bigram_mask & prev_is_s).sum()
        per_subj_counts[s] = cnt
        print(f"  SUBJ_{s - subj_start} (token {s}): {cnt:>5d} positions")

    for mod in args.modules:
        if mod not in ci_full:
            print(f"\n[skip] {mod} not found in CI dict")
            continue
        ci_m = ci_full[mod]
        n_seq, S, C = ci_m.shape
        print(f"\n=== {mod}  (C = {C}) ===")
        # For each SUBJ, compute mean ci per component on bigram positions where prev=that SUBJ
        per_subj_means: dict[int, np.ndarray] = {}
        for s in range(subj_start, subj_start + n_subj):
            prev_is_s = np.zeros_like(anns_all, dtype=bool)
            prev_is_s[:, 1:] = seqs_arr[:, :-1] == s
            mask = bigram_mask & prev_is_s
            if mask.sum() < 20:
                continue
            sel = ci_m[mask]  # (n, C)
            per_subj_means[s] = sel.mean(axis=0)

        # Print top-5 per-SUBJ components and which SUBJs they prefer
        # Component selectivity: max_subj(mean) / second_max(mean)
        means_matrix = np.stack([per_subj_means[s] for s in sorted(per_subj_means)], axis=0)  # (n_subj, C)
        # For each component, find the dominant SUBJ
        dom_subj = np.argmax(means_matrix, axis=0)  # (C,)
        dom_val = np.max(means_matrix, axis=0)      # (C,)
        # Selectivity: ratio of max to mean-of-others
        sorted_subj = sorted(per_subj_means.keys())
        other_means = (means_matrix.sum(axis=0) - dom_val) / max(1, means_matrix.shape[0] - 1)
        selectivity = dom_val / np.maximum(other_means, 1e-6)

        # Top per-SUBJ components
        print(f"  Top component for each SUBJ (highest mean activation when that SUBJ is the trigger):")
        for i, s in enumerate(sorted_subj):
            top_c = int(np.argmax(means_matrix[i]))
            top_val = means_matrix[i, top_c]
            # the selectivity for this component
            print(f"    SUBJ_{s - subj_start}: top component {top_c:>3d} (mean ci = {top_val:.3f}, dominant for SUBJ_{dom_subj[top_c]}, sel = {selectivity[top_c]:.2f})")

        # Most SUBJ-selective components overall
        sel_idx = np.argsort(-selectivity)[:5]
        print(f"  Most SUBJ-selective components (sel = max/mean-of-others):")
        for c in sel_idx:
            print(f"    comp {c:>3d}: dominant for SUBJ_{dom_subj[c]} (mean={means_matrix[dom_subj[c], c]:.3f}, sel={selectivity[c]:.2f}); per-SUBJ: " +
                  ", ".join(f"S{s-subj_start}={means_matrix[i,c]:.2f}" for i, s in enumerate(sorted_subj)))


if __name__ == "__main__":
    main()
