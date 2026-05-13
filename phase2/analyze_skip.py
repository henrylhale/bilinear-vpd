"""For each skip-trigram rule (LOC, ADJ) -> CONN, find which components
fire most when that specific rule is the trigger at this position.

Run from repo root:
   PYTHONPATH=. ~/miniconda3/envs/vpd/bin/python -m phase2.analyze_skip \
       --target runs/v16_chan_lr3e3 --vpd runs/vpd_v16_B --n_sequences 2048
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from phase1.config import default_dgp
from phase1.data import ANN_SKIP, DGP
from phase2.run_decomposition import build_component_model, load_target_model
from phase2.config import default_vpd_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--vpd", required=True)
    parser.add_argument("--n_sequences", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=[
            "blocks.0.attn.q1_proj",
            "blocks.0.attn.k1_proj",
            "blocks.0.attn.v_proj",
            "blocks.0.attn.o_proj",
            "blocks.1.mlp.w_m",
            "unembed",
        ],
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
    rng = np.random.default_rng(202)
    print(f"Sampling {args.n_sequences} sequences (skip-trigram is rare, need a lot)…")
    seqs, anns = [], []
    for _ in range(args.n_sequences):
        t, a, _ = dgp.sample_sequence(rng)
        seqs.append(t); anns.append(a)
    toks_all = torch.from_numpy(np.stack(seqs))
    seqs_arr = np.stack(seqs)
    anns_arr = np.stack(anns)

    print("Forward pass…")
    ci_collected: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        for s in range(0, toks_all.shape[0], 256):
            e = min(s + 256, toks_all.shape[0])
            out = cm(toks_all[s:e].to(device), cache_type="input")
            ci = cm.calc_causal_importances(out.cache, sampling="continuous")
            for mod, g in ci.upper_leaky.items():
                if mod in args.modules:
                    ci_collected[mod].append(g.cpu())
    ci_full = {m: torch.cat(parts, dim=0).numpy() for m, parts in ci_collected.items()}

    # For each skip-firing position, identify the LOC,ADJ pair that triggered the rule
    skip_mask = anns_arr == ANN_SKIP
    print(f"\nTotal skip-firing positions: {skip_mask.sum()}")

    # The trigger ADJ is tokens[t-1] at the skip-firing position; the LOC comes from
    # the most-recent lookback. Use DGP._find_skip_loc to recover it.
    rule_keys: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros_like(anns_arr, dtype=bool))
    bs_, ts_ = np.nonzero(skip_mask)
    for bi, ti in zip(bs_.tolist(), ts_.tolist()):
        seq = seqs_arr[bi]
        prev_adj = int(seq[ti - 1])
        loc = dgp._find_skip_loc(seq, ti)
        if loc < 0:
            continue
        rule_keys[(loc, prev_adj)][bi, ti] = True

    print(f"Distinct (LOC, ADJ) rule firings observed: {len(rule_keys)}")
    rule_list = sorted(rule_keys.keys(), key=lambda k: -rule_keys[k].sum())
    for loc, adj in rule_list[:8]:
        cnt = rule_keys[(loc, adj)].sum()
        conn = dgp.rules.skip_rules.get((loc, adj), -1)
        print(f"  (LOC_{loc - dgp.vocab.loc_start}, ADJ_{adj - dgp.vocab.adj_start}) -> CONN_{conn - dgp.vocab.conn_start}: {cnt} firings")

    # For each module, compute per-rule mean importance, find rule-selective components
    for mod in args.modules:
        if mod not in ci_full:
            continue
        ci_m = ci_full[mod]
        C = ci_m.shape[2]
        print(f"\n=== {mod}  (C = {C}) — top rule-selective components ===")
        # Per-rule means (require ≥ 10 firings to include in contrast)
        per_rule: dict[tuple[int, int], np.ndarray] = {}
        for rule, mask in rule_keys.items():
            if mask.sum() >= 10:
                per_rule[rule] = ci_m[mask].mean(axis=0)
        if not per_rule:
            continue
        means_matrix = np.stack([per_rule[r] for r in per_rule], axis=0)  # (n_rules, C)
        rule_idx = {r: i for i, r in enumerate(per_rule)}
        # Selectivity: max / mean-of-others
        dom = np.argmax(means_matrix, axis=0)
        dom_val = np.max(means_matrix, axis=0)
        mean_others = (means_matrix.sum(axis=0) - dom_val) / max(1, means_matrix.shape[0] - 1)
        sel = dom_val / np.maximum(mean_others, 1e-3)
        top_c = np.argsort(-sel)[:8]
        print(f"  {'comp':>4s}  {'dominant rule':<22s}  {'mean':>5s}  {'others_avg':>10s}  {'sel':>5s}")
        for c in top_c:
            rules = list(per_rule.keys())
            dom_rule = rules[dom[c]]
            loc, adj = dom_rule
            conn = dgp.rules.skip_rules[dom_rule]
            label = f"L{loc - dgp.vocab.loc_start}+A{adj - dgp.vocab.adj_start}->C{conn - dgp.vocab.conn_start}"
            print(f"  {c:>4d}  {label:<22s}  {dom_val[c]:>5.2f}  {mean_others[c]:>10.3f}  {sel[c]:>5.2f}")


if __name__ == "__main__":
    main()
