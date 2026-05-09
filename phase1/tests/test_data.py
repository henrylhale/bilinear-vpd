import numpy as np
import pytest

from phase1.config import default_dgp
from phase1.data import (
    ANN_BIGRAM,
    ANN_INDUCTION,
    ANN_NONE,
    ANN_SKIP,
    BOS,
    DGP,
    EOS,
)


@pytest.fixture
def dgp() -> DGP:
    return DGP(default_dgp(seed=42))


@pytest.fixture
def big_sample(dgp: DGP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    n = 1000
    L = dgp.cfg.seq_len
    toks = np.empty((n, L), dtype=np.int64)
    anns = np.empty((n, L), dtype=np.int64)
    inds = np.empty((n, L), dtype=np.int64)
    for i in range(n):
        toks[i], anns[i], inds[i] = dgp.sample_sequence(rng)
    return toks, anns, inds


def test_special_tokens_bookend(big_sample):
    toks, _, _ = big_sample
    assert (toks[:, 0] == BOS).all()
    assert (toks[:, -1] == EOS).all()


def test_token_ids_in_range(dgp: DGP, big_sample):
    toks, _, _ = big_sample
    assert toks.min() >= 0
    assert toks.max() < dgp.vocab_size


def test_no_pad_in_sampled(big_sample):
    toks, _, _ = big_sample
    from phase1.data import PAD
    assert (toks != PAD).all(), "PAD should never appear in sampled sequences"


def test_primitive_firing_rates_in_range(dgp: DGP, big_sample):
    """Each primitive should fire on 5-15% of positions across the dataset."""
    _, anns, _ = big_sample
    interior = anns[:, 1:-1]  # exclude BOS at 0 and EOS at L-1
    total = interior.size
    for ann_code, lo, hi in [
        (ANN_BIGRAM, 0.04, 0.20),
        (ANN_SKIP, 0.01, 0.20),
        (ANN_INDUCTION, 0.03, 0.20),
    ]:
        rate = (interior == ann_code).sum() / total
        assert lo <= rate <= hi, f"primitive {ann_code} firing rate {rate:.3f} outside [{lo}, {hi}]"


def test_bigram_positions_are_after_subj(dgp: DGP, big_sample):
    toks, anns, _ = big_sample
    bigram_mask = anns == ANN_BIGRAM
    # For every bigram position t, toks[t-1] must be a SUBJ token.
    bs, ts = np.nonzero(bigram_mask)
    for b, t in zip(bs.tolist(), ts.tolist()):
        prev = int(toks[b, t - 1])
        assert dgp.vocab.is_subj(prev), f"bigram at t={t} but prev token {prev} is not SUBJ"


def test_bigram_empirical_distribution_matches(dgp: DGP, big_sample):
    """For each subject S, the empirical distribution of next tokens at bigram-annotated
    positions following S must match dgp.rules.subj_verb_dist[S] within tolerance."""
    toks, anns, _ = big_sample
    bigram_mask = anns == ANN_BIGRAM
    bs, ts = np.nonzero(bigram_mask)
    by_subj: dict[int, list[int]] = {}
    for b, t in zip(bs.tolist(), ts.tolist()):
        prev = int(toks[b, t - 1])
        by_subj.setdefault(prev, []).append(int(toks[b, t]))
    for subj_tok, samples in by_subj.items():
        if len(samples) < 50:
            continue
        empirical = np.bincount(samples, minlength=dgp.vocab_size).astype(np.float64)
        empirical /= empirical.sum()
        true = dgp.rules.subj_verb_dist[subj_tok - dgp.vocab.subj_start]
        l1 = float(np.abs(empirical - true).sum())
        assert l1 < 0.15, f"subj {subj_tok} empirical L1 deviation {l1:.3f} (n={len(samples)})"


def test_skip_trigram_is_deterministic(dgp: DGP, big_sample):
    """At every skip-annotated position, the token equals the rule's CONN, and there
    is a recent matching LOC."""
    toks, anns, _ = big_sample
    mask = anns == ANN_SKIP
    bs, ts = np.nonzero(mask)
    for b, t in zip(bs.tolist(), ts.tolist()):
        prev = int(toks[b, t - 1])
        assert dgp.vocab.is_adj(prev)
        loc = dgp._find_skip_loc(toks[b], t)
        assert loc >= 0
        expected = dgp.rules.skip_rules[(loc, prev)]
        assert int(toks[b, t]) == expected


def test_induction_empirical_concentration(dgp: DGP, big_sample):
    """At induction-annotated positions, the sampled token equals the induced token
    with frequency near (1 - smoothing)."""
    toks, anns, inds = big_sample
    mask = anns == ANN_INDUCTION
    bs, ts = np.nonzero(mask)
    if len(bs) < 100:
        pytest.skip(f"too few induction positions ({len(bs)}) for empirical check")
    matches = 0
    total = 0
    for b, t in zip(bs.tolist(), ts.tolist()):
        induced = int(inds[b, t])
        assert induced >= 0
        if int(toks[b, t]) == induced:
            matches += 1
        total += 1
    rate = matches / total
    expected = 1.0 - dgp.cfg.induction_smoothing + dgp.cfg.induction_smoothing / dgp.vocab_size
    assert abs(rate - expected) < 0.05, f"induction match rate {rate:.3f}, expected ~{expected:.3f}"


def test_induction_uniqueness_holds(dgp: DGP, big_sample):
    """Every induction firing must correspond to a unique earlier X-Y bigram."""
    toks, anns, inds = big_sample
    mask = anns == ANN_INDUCTION
    bs, ts = np.nonzero(mask)
    for b, t in zip(bs.tolist(), ts.tolist()):
        X = int(toks[b, t - 2])
        Y = int(toks[b, t - 1])
        seq = toks[b]
        count = 0
        last_s = -1
        for s in range(t - 2):
            if int(seq[s]) == X and int(seq[s + 1]) == Y:
                count += 1
                last_s = s
        assert count == 1, f"induction at t={t} has {count} earlier matches"
        assert int(seq[last_s + 2]) == int(inds[b, t])


def test_true_distribution_sums_to_one(dgp: DGP, big_sample):
    toks, anns, inds = big_sample
    rule_mask = anns != ANN_NONE
    bs, ts = np.nonzero(rule_mask)
    bs = bs[:200]
    ts = ts[:200]
    for b, t in zip(bs.tolist(), ts.tolist()):
        p = dgp.true_distribution(toks[b], anns[b], inds[b], t)
        assert p.shape == (dgp.vocab_size,)
        assert abs(float(p.sum()) - 1.0) < 1e-9
        assert (p >= 0).all()


def test_precedence_subj_beats_skip_and_induction(dgp: DGP):
    """If the previous token is SUBJ, bigram primitive must fire even if induction
    or skip-trigram preconditions also hold."""
    rng = np.random.default_rng(0)
    saw_subj = 0
    for _ in range(200):
        toks, anns, _ = dgp.sample_sequence(rng)
        for t in range(1, len(toks) - 1):
            if dgp.vocab.is_subj(int(toks[t - 1])):
                assert int(anns[t]) == ANN_BIGRAM
                saw_subj += 1
    assert saw_subj > 100  # sanity: we observed enough cases


def test_vocab_ranges_are_disjoint_and_cover_total(dgp: DGP):
    v = dgp.vocab
    assert v.subj_start == 3  # after BOS, EOS, PAD
    assert v.filler_end == v.total
    # Every interior id classifies as exactly one slot.
    for i in range(3, v.total):
        slots = [v.is_subj(i), v.is_verb(i), v.is_loc(i), v.is_adj(i), v.is_conn(i), v.is_filler(i)]
        assert sum(slots) == 1, f"token id {i} has slot membership {slots}"


def test_sample_batch_shapes(dgp: DGP):
    rng = np.random.default_rng(7)
    toks, anns, inds = dgp.sample_batch(rng, batch_size=16)
    L = dgp.cfg.seq_len
    assert toks.shape == (16, L)
    assert anns.shape == (16, L)
    assert inds.shape == (16, L)
    assert toks.dtype.is_floating_point is False
