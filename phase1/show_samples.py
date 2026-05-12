"""Print a handful of DGP samples as tokens with slot + primitive annotations.

Run from repo root:
    PYTHONPATH=. ~/miniconda3/envs/bilinear-vpd/bin/python -m phase1.show_samples
    PYTHONPATH=. ~/miniconda3/envs/bilinear-vpd/bin/python -m phase1.show_samples \
        --n 5 --seed 42 --show-rules
"""
import argparse

import numpy as np

from phase1.config import default_dgp
from phase1.data import (
    ANN_BIGRAM,
    ANN_INDUCTION,
    ANN_NONE,
    ANN_SKIP,
    BOS,
    DGP,
    EOS,
    PAD,
)


def slot_name(v, t: int) -> str:
    if t == BOS:
        return "BOS"
    if t == EOS:
        return "EOS"
    if t == PAD:
        return "PAD"
    if v.is_subj(t):
        return f"SUBJ_{t - v.subj_start}"
    if v.is_verb(t):
        return f"VERB_{t - v.verb_start}"
    if v.is_loc(t):
        return f"LOC_{t - v.loc_start}"
    if v.is_adj(t):
        return f"ADJ_{t - v.adj_start}"
    if v.is_conn(t):
        return f"CONN_{t - v.conn_start}"
    if v.is_filler(t):
        return f"FIL_{t - v.filler_start}"
    return f"?_{t}"


ANN_LABEL = {
    ANN_NONE: "default",
    ANN_BIGRAM: "BIGRAM",
    ANN_SKIP: "SKIP",
    ANN_INDUCTION: "INDUCT",
}


def print_rules(dgp: DGP) -> None:
    v = dgp.vocab
    print("# Vocab slot ranges (token IDs)")
    print(f"  specials  [BOS=0, EOS=1, PAD=2]")
    print(f"  SUBJ      {v.subj_start}..{v.subj_end - 1}    ({v.sizes.n_subj} tokens)")
    print(f"  VERB      {v.verb_start}..{v.verb_end - 1}   ({v.sizes.n_verb} tokens)")
    print(f"  LOC       {v.loc_start}..{v.loc_end - 1}    ({v.sizes.n_loc} tokens)")
    print(f"  ADJ       {v.adj_start}..{v.adj_end - 1}    ({v.sizes.n_adj} tokens)")
    print(f"  CONN      {v.conn_start}..{v.conn_end - 1}    ({v.sizes.n_conn} tokens)")
    print(f"  FILLER    {v.filler_start}..{v.filler_end - 1}   ({v.sizes.n_filler} tokens)")
    print(f"  vocab_size = {v.total}\n")

    print("# Bigram rule:  SUBJ -> top-3 verb distribution (probs 0.7, 0.2, 0.1)")
    for s in range(v.subj_start, v.subj_end):
        p = dgp.rules.subj_verb_dist[s - v.subj_start]
        top = np.argsort(-p)[:3]
        items = ", ".join(f"{slot_name(v, int(t))}({p[t]:.2f})" for t in top)
        print(f"  {slot_name(v, s):>9s}  ->  {items}")
    print()

    print(f"# Skip-trigram rules:  (LOC, ADJ) -> CONN  [{len(dgp.rules.skip_rules)} pairs]")
    for (loc, adj), conn in sorted(dgp.rules.skip_rules.items()):
        print(f"  ({slot_name(v, loc):>6s}, {slot_name(v, adj):>6s})  ->  {slot_name(v, conn)}")
    print()


def annotate_sequence(dgp: DGP, tokens: np.ndarray, ann: np.ndarray, induced: np.ndarray) -> None:
    v = dgp.vocab
    print(f"  {'pos':>3s}  {'tok':>3s}  {'slot':<10s}  {'primitive':<8s}  notes")
    for t in range(len(tokens)):
        a = int(ann[t])
        note = ""
        if a == ANN_BIGRAM:
            prev = int(tokens[t - 1])
            note = f"prev was {slot_name(v, prev)}; sampled from its verb dist"
        elif a == ANN_SKIP:
            loc = dgp._find_skip_loc(tokens, t)
            prev = int(tokens[t - 1])
            note = (
                f"prev ADJ {slot_name(v, prev)} + recent LOC {slot_name(v, loc)} "
                f"-> deterministic CONN"
            )
        elif a == ANN_INDUCTION:
            x = int(tokens[t - 1])
            note = (
                f"prev token {slot_name(v, x)} occurred uniquely before; "
                f"past follower -> {slot_name(v, int(induced[t]))}"
            )
        elif a == ANN_NONE and t > 0:
            note = "default unigram"
        print(
            f"  {t:>3d}  {int(tokens[t]):>3d}  {slot_name(v, int(tokens[t])):<10s}  "
            f"{ANN_LABEL[a]:<8s}  {note}"
        )


def print_firing_stats(dgp: DGP, rng: np.random.Generator, n_seqs: int) -> None:
    toks, anns, inds = dgp.sample_batch(rng, n_seqs)
    anns_np = anns.numpy()
    interior = anns_np[:, 1:-1]
    total = interior.size
    print(f"# Firing-rate stats over {n_seqs} sequences ({total} interior positions)")
    for code, name in [
        (ANN_NONE, "default"),
        (ANN_BIGRAM, "BIGRAM"),
        (ANN_SKIP, "SKIP"),
        (ANN_INDUCTION, "INDUCT"),
    ]:
        n = int((interior == code).sum())
        print(f"  {name:<8s}  {n:>5d}  ({100 * n / total:5.2f}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2, help="number of sample sequences to print")
    parser.add_argument("--seed", type=int, default=7, help="rng seed (for sampling, not for DGP rules)")
    parser.add_argument("--show-rules", action="store_true", help="print rule tables before samples")
    parser.add_argument("--stats", action="store_true", help="print firing-rate stats over 500 sequences")
    parser.add_argument("--dgp-seed", type=int, default=0, help="seed for DGP rule construction")
    parser.add_argument("--seq-len", type=int, default=None, help="override seq_len")
    args = parser.parse_args()

    cfg = default_dgp(seed=args.dgp_seed)
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    dgp = DGP(cfg)

    if args.show_rules:
        print_rules(dgp)

    if args.stats:
        print_firing_stats(dgp, np.random.default_rng(args.seed), n_seqs=500)

    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        print(f"# Sample {i + 1}/{args.n}")
        tokens, ann, induced = dgp.sample_sequence(rng)
        annotate_sequence(dgp, tokens, ann, induced)
        print()


if __name__ == "__main__":
    main()
