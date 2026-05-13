"""Drill into induction-firing positions. For each induction position, the
trigger is `tokens[t-1]` (a non-filler token from SUBJ/VERB/LOC/ADJ/CONN).
We bucket induction firings by *slot* of the trigger and look at which
components in Layer 1 attention fire selectively for each slot.

   PYTHONPATH=. ~/miniconda3/envs/vpd/bin/python -m phase2.analyze_induction \
       --target runs/v16_chan_lr3e3 --vpd runs/vpd_v16_B --n_sequences 1024
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from phase1.config import default_dgp
from phase1.data import ANN_INDUCTION, DGP
from phase2.run_decomposition import build_component_model, load_target_model
from phase2.config import default_vpd_config


SLOT_NAMES = ["SUBJ", "VERB", "LOC", "ADJ", "CONN"]


def slot_of_token(vocab, t: int) -> str | None:
    if vocab.is_subj(t): return "SUBJ"
    if vocab.is_verb(t): return "VERB"
    if vocab.is_loc(t):  return "LOC"
    if vocab.is_adj(t):  return "ADJ"
    if vocab.is_conn(t): return "CONN"
    return None


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
            "blocks.0.attn.k1_proj",
            "blocks.0.attn.k2_proj",
            "blocks.1.attn.k1_proj",
            "blocks.1.attn.k2_proj",
            "blocks.1.attn.v_proj",
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
    rng = np.random.default_rng(101)
    print(f"Sampling {args.n_sequences} sequences…")
    seqs, anns, inducs = [], [], []
    for _ in range(args.n_sequences):
        t, a, ind = dgp.sample_sequence(rng)
        seqs.append(t); anns.append(a); inducs.append(ind)
    toks_all = torch.from_numpy(np.stack(seqs))
    seqs_arr = np.stack(seqs)
    anns_arr = np.stack(anns)
    inds_arr = np.stack(inducs)

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

    # Build slot masks indexed by trigger-token slot at position t (i.e., based on tokens[t-1])
    induct_mask = anns_arr == ANN_INDUCTION
    # For each induction position, slot of trigger token tokens[t-1]
    # We'll compute mean ci per slot.
    # First derive slot-of-prev-token grid
    print(f"\nTotal induction positions: {induct_mask.sum()}")
    print("Breakdown by trigger-token slot (= slot of tokens[t-1]):")
    slot_masks: dict[str, np.ndarray] = {}
    for slot in SLOT_NAMES:
        prev_in_slot = np.zeros_like(anns_arr, dtype=bool)
        for t in range(1, seqs_arr.shape[1]):
            tok = seqs_arr[:, t - 1]
            if slot == "SUBJ":  s_in = (tok >= dgp.vocab.subj_start) & (tok < dgp.vocab.subj_end)
            elif slot == "VERB": s_in = (tok >= dgp.vocab.verb_start) & (tok < dgp.vocab.verb_end)
            elif slot == "LOC":  s_in = (tok >= dgp.vocab.loc_start)  & (tok < dgp.vocab.loc_end)
            elif slot == "ADJ":  s_in = (tok >= dgp.vocab.adj_start)  & (tok < dgp.vocab.adj_end)
            elif slot == "CONN": s_in = (tok >= dgp.vocab.conn_start) & (tok < dgp.vocab.conn_end)
            prev_in_slot[:, t] = s_in
        slot_mask = induct_mask & prev_in_slot
        slot_masks[slot] = slot_mask
        print(f"  trigger is {slot:<5s}: {slot_mask.sum():>5d} positions")

    # Also: slot of induced token (the predicted Y)
    print("\nBreakdown by induced-token slot:")
    induced_slot_masks: dict[str, np.ndarray] = {}
    for slot in SLOT_NAMES:
        ind_in_slot = np.zeros_like(anns_arr, dtype=bool)
        if slot == "SUBJ":  s_in = (inds_arr >= dgp.vocab.subj_start) & (inds_arr < dgp.vocab.subj_end)
        elif slot == "VERB": s_in = (inds_arr >= dgp.vocab.verb_start) & (inds_arr < dgp.vocab.verb_end)
        elif slot == "LOC":  s_in = (inds_arr >= dgp.vocab.loc_start)  & (inds_arr < dgp.vocab.loc_end)
        elif slot == "ADJ":  s_in = (inds_arr >= dgp.vocab.adj_start)  & (inds_arr < dgp.vocab.adj_end)
        elif slot == "CONN": s_in = (inds_arr >= dgp.vocab.conn_start) & (inds_arr < dgp.vocab.conn_end)
        else:                s_in = np.zeros_like(inds_arr, dtype=bool)
        ind_in_slot = induct_mask & s_in
        # Filler-induced (when induced token is a FILLER) — also relevant
        induced_slot_masks[slot] = ind_in_slot
        print(f"  induced is {slot:<5s}: {ind_in_slot.sum():>5d} positions")
    # And filler
    is_filler = (inds_arr >= dgp.vocab.filler_start) & (inds_arr < dgp.vocab.filler_end)
    print(f"  induced is FIL  : {(induct_mask & is_filler).sum():>5d} positions")

    # Determine which slots have enough data
    nonempty_slots = [s for s in SLOT_NAMES if slot_masks[s].sum() >= 20]
    print(f"\nUsing non-empty trigger slots only: {nonempty_slots}")

    for mod in args.modules:
        if mod not in ci_full:
            print(f"\n[skip] {mod}")
            continue
        ci_m = ci_full[mod]  # (N, S, C)
        C = ci_m.shape[2]
        print(f"\n=== {mod}  (C = {C}) — induction trigger slot breakdown ===")
        print(f"  {'comp':>4s}  " + "  ".join(f"{s:>5s}" for s in nonempty_slots) + "  sel(max/min)")
        slot_means = {s: ci_m[slot_masks[s]].mean(axis=0) for s in nonempty_slots}
        max_vals = np.max(np.stack([slot_means[s] for s in nonempty_slots], axis=0), axis=0)
        min_vals = np.min(np.stack([slot_means[s] for s in nonempty_slots], axis=0), axis=0)
        sel = max_vals / np.maximum(min_vals, 1e-3)
        top_c = np.argsort(-sel)[:8]
        for c in top_c:
            print(f"  {c:>4d}  " + "  ".join(f"{slot_means[s][c]:>5.2f}" for s in nonempty_slots) + f"  {sel[c]:>7.2f}")


if __name__ == "__main__":
    main()
