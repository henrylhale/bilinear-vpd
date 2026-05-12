from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from phase1.config import DGPConfig, VocabSizes


# Annotation codes
ANN_NONE = 0
ANN_BIGRAM = 1
ANN_SKIP = 2
ANN_INDUCTION = 3
N_ANN = 4
ANN_NAMES = {ANN_NONE: "none", ANN_BIGRAM: "bigram", ANN_SKIP: "skip", ANN_INDUCTION: "induction"}

# Special token ids
BOS = 0
EOS = 1
PAD = 2


@dataclass
class Vocab:
    sizes: VocabSizes

    @property
    def subj_start(self) -> int: return 3
    @property
    def subj_end(self) -> int: return self.subj_start + self.sizes.n_subj
    @property
    def verb_start(self) -> int: return self.subj_end
    @property
    def verb_end(self) -> int: return self.verb_start + self.sizes.n_verb
    @property
    def loc_start(self) -> int: return self.verb_end
    @property
    def loc_end(self) -> int: return self.loc_start + self.sizes.n_loc
    @property
    def adj_start(self) -> int: return self.loc_end
    @property
    def adj_end(self) -> int: return self.adj_start + self.sizes.n_adj
    @property
    def conn_start(self) -> int: return self.adj_end
    @property
    def conn_end(self) -> int: return self.conn_start + self.sizes.n_conn
    @property
    def filler_start(self) -> int: return self.conn_end
    @property
    def filler_end(self) -> int: return self.filler_start + self.sizes.n_filler

    @property
    def total(self) -> int: return self.sizes.total

    def is_subj(self, t: int) -> bool: return self.subj_start <= t < self.subj_end
    def is_verb(self, t: int) -> bool: return self.verb_start <= t < self.verb_end
    def is_loc(self, t: int) -> bool: return self.loc_start <= t < self.loc_end
    def is_adj(self, t: int) -> bool: return self.adj_start <= t < self.adj_end
    def is_conn(self, t: int) -> bool: return self.conn_start <= t < self.conn_end
    def is_filler(self, t: int) -> bool: return self.filler_start <= t < self.filler_end


@dataclass
class RuleTables:
    subj_verb_dist: Float[np.ndarray, "n_subj vocab"]
    skip_rules: dict[tuple[int, int], int]
    default_dist: Float[np.ndarray, "vocab"]


def _build_default_dist(vocab: Vocab, cfg: DGPConfig) -> np.ndarray:
    dist = np.zeros(vocab.total, dtype=np.float64)
    dist[vocab.filler_start:vocab.filler_end] = cfg.default_p_filler / vocab.sizes.n_filler
    dist[vocab.subj_start:vocab.subj_end] = cfg.default_p_subj / vocab.sizes.n_subj
    dist[vocab.loc_start:vocab.loc_end] = cfg.default_p_loc / vocab.sizes.n_loc
    dist[vocab.adj_start:vocab.adj_end] = cfg.default_p_adj / vocab.sizes.n_adj
    dist[vocab.verb_start:vocab.verb_end] = cfg.default_p_verb / vocab.sizes.n_verb
    dist[vocab.conn_start:vocab.conn_end] = cfg.default_p_conn / vocab.sizes.n_conn
    s = float(dist.sum())
    assert abs(s - 1.0) < 1e-9, f"default unigram sums to {s}, not 1"
    return dist


def _build_subj_verb_dist(vocab: Vocab, cfg: DGPConfig, rng: np.random.Generator) -> np.ndarray:
    n_subj = vocab.sizes.n_subj
    n_verb = vocab.sizes.n_verb
    dist = np.zeros((n_subj, vocab.total), dtype=np.float64)
    assert cfg.bigram_top_k <= n_verb
    assert len(cfg.bigram_probs) == cfg.bigram_top_k
    assert abs(sum(cfg.bigram_probs) - 1.0) < 1e-9
    for i in range(n_subj):
        verb_indices = rng.choice(n_verb, size=cfg.bigram_top_k, replace=False)
        for k, vi in enumerate(verb_indices):
            tok = vocab.verb_start + int(vi)
            dist[i, tok] = cfg.bigram_probs[k]
    return dist


def _build_skip_rules(
    vocab: Vocab, cfg: DGPConfig, rng: np.random.Generator
) -> dict[tuple[int, int], int]:
    n_loc = vocab.sizes.n_loc
    n_adj = vocab.sizes.n_adj
    n_conn = vocab.sizes.n_conn
    K = cfg.skip_trigram_n_pairs
    assert K <= n_loc * n_adj
    rules: dict[tuple[int, int], int] = {}
    flat = rng.choice(n_loc * n_adj, size=K, replace=False)
    for j, idx in enumerate(flat):
        li = int(idx) // n_adj
        ai = int(idx) % n_adj
        loc_tok = vocab.loc_start + li
        adj_tok = vocab.adj_start + ai
        conn_tok = vocab.conn_start + (j % n_conn)
        rules[(loc_tok, adj_tok)] = conn_tok
    return rules


class DGP:
    def __init__(self, cfg: DGPConfig):
        self.cfg = cfg
        self.vocab = Vocab(sizes=cfg.vocab)
        rng = np.random.default_rng(cfg.seed)
        self.rules = RuleTables(
            subj_verb_dist=_build_subj_verb_dist(self.vocab, cfg, rng),
            skip_rules=_build_skip_rules(self.vocab, cfg, rng),
            default_dist=_build_default_dist(self.vocab, cfg),
        )
        self.vocab_size = self.vocab.total
        # Precomputed cumulative distributions for fast searchsorted sampling.
        self._default_cdf = np.cumsum(self.rules.default_dist)
        self._subj_verb_cdfs = np.cumsum(self.rules.subj_verb_dist, axis=1)

    def _sample_default(self, rng: np.random.Generator) -> int:
        u = float(rng.random())
        return int(np.searchsorted(self._default_cdf, u))

    def _sample_subj_verb(self, rng: np.random.Generator, subj_idx: int) -> int:
        u = float(rng.random())
        return int(np.searchsorted(self._subj_verb_cdfs[subj_idx], u))

    def _sample_induced(self, rng: np.random.Generator, induced_token: int) -> int:
        s = self.cfg.induction_smoothing
        if rng.random() < (1.0 - s):
            return induced_token
        n_special = self.vocab.sizes.n_special
        return int(rng.integers(n_special, self.vocab_size))

    def _induced_dist(self, induced_token: int) -> np.ndarray:
        v = self.vocab_size
        n_special = self.vocab.sizes.n_special
        n_content = v - n_special
        s = self.cfg.induction_smoothing
        p = np.zeros(v, dtype=np.float64)
        p[n_special:] = s / n_content
        p[induced_token] += 1.0 - s
        return p

    def _find_skip_loc(self, tokens: np.ndarray, t: int) -> int:
        """Most recent LOC L in tokens[t-1-max_skip : t-1] such that (L, tokens[t-1]) is a rule. -1 if none."""
        max_skip = self.cfg.skip_trigram_max_skip
        prev_adj = int(tokens[t - 1])
        lookback_start = max(0, t - 1 - max_skip)
        for s in range(t - 2, lookback_start - 1, -1):
            tok = int(tokens[s])
            if self.vocab.is_loc(tok) and (tok, prev_adj) in self.rules.skip_rules:
                return tok
        return -1

    def sample_sequence(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = self.cfg.seq_len
        tokens = np.zeros(L, dtype=np.int64)
        ann = np.zeros(L, dtype=np.int64)
        induced = -np.ones(L, dtype=np.int64)
        tokens[0] = BOS
        # Classic induction history: token X -> list of followers Y.
        # Registered for each position s where the bigram (tokens[s], tokens[s+1])
        # has fully entered the past at the start of this iteration.
        token_followers: dict[int, list[int]] = {}
        for t in range(1, L - 1):
            # The most recent s for which tokens[s+1] is strictly before position t-1
            # is s = t - 3. At iteration t (with t >= 3) we register that mapping.
            if t >= 3:
                x_past = int(tokens[t - 3])
                token_followers.setdefault(x_past, []).append(int(tokens[t - 2]))

            prev = int(tokens[t - 1])
            if self.vocab.is_subj(prev):
                tokens[t] = self._sample_subj_verb(rng, prev - self.vocab.subj_start)
                ann[t] = ANN_BIGRAM
                continue
            if self.vocab.is_adj(prev):
                loc = self._find_skip_loc(tokens, t)
                if loc >= 0:
                    tokens[t] = self.rules.skip_rules[(loc, prev)]
                    ann[t] = ANN_SKIP
                    continue
            if not self.vocab.is_filler(prev):
                followers = token_followers.get(prev)
                if followers is not None and len(followers) == 1:
                    induced_tok = followers[0]
                    tokens[t] = self._sample_induced(rng, induced_tok)
                    ann[t] = ANN_INDUCTION
                    induced[t] = induced_tok
                    continue
            tokens[t] = self._sample_default(rng)
            ann[t] = ANN_NONE
        tokens[L - 1] = EOS
        return tokens, ann, induced

    def sample_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[Int[Tensor, "B S"], Int[Tensor, "B S"], Int[Tensor, "B S"]]:
        toks = np.empty((batch_size, self.cfg.seq_len), dtype=np.int64)
        anns = np.empty((batch_size, self.cfg.seq_len), dtype=np.int64)
        inds = np.empty((batch_size, self.cfg.seq_len), dtype=np.int64)
        for i in range(batch_size):
            toks[i], anns[i], inds[i] = self.sample_sequence(rng)
        return torch.from_numpy(toks), torch.from_numpy(anns), torch.from_numpy(inds)

    def true_distribution(
        self, tokens: np.ndarray, ann: np.ndarray, induced: np.ndarray, t: int
    ) -> np.ndarray:
        """Ground-truth next-token distribution at position t. Caller must ensure ann[t] != ANN_NONE."""
        a = int(ann[t])
        if a == ANN_BIGRAM:
            prev = int(tokens[t - 1])
            assert self.vocab.is_subj(prev)
            return self.rules.subj_verb_dist[prev - self.vocab.subj_start]
        if a == ANN_SKIP:
            loc = self._find_skip_loc(tokens, t)
            assert loc >= 0, f"skip ann at t={t} but no matching rule"
            conn = self.rules.skip_rules[(loc, int(tokens[t - 1]))]
            p = np.zeros(self.vocab_size, dtype=np.float64)
            p[conn] = 1.0
            return p
        if a == ANN_INDUCTION:
            induced_tok = int(induced[t])
            assert induced_tok >= 0
            return self._induced_dist(induced_tok)
        raise AssertionError(f"true_distribution called for ANN_NONE at t={t}")


class DGPIterableDataset(IterableDataset):
    """Yields full (tokens, anns, induced) batches; designed for use with
    DataLoader(num_workers=K, batch_size=None) so each worker process produces
    batches in parallel. Each worker constructs its own DGP from cfg (rule
    tables are deterministic in cfg.seed), and uses a worker-distinct rng seed
    so the streams don't collide."""

    def __init__(self, cfg: DGPConfig, batch_size: int, base_seed: int):
        self.cfg = cfg
        self.batch_size = batch_size
        self.base_seed = base_seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi is not None else 0
        dgp = DGP(self.cfg)
        rng = np.random.default_rng(self.base_seed * 1_000_003 + worker_id * 7919 + 1)
        while True:
            yield dgp.sample_batch(rng, self.batch_size)


def make_train_loader(cfg: DGPConfig, batch_size: int, base_seed: int, num_workers: int) -> DataLoader:
    ds = DGPIterableDataset(cfg, batch_size, base_seed)
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
