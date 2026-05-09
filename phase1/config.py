from dataclasses import dataclass, field


@dataclass
class VocabSizes:
    n_subj: int
    n_verb: int
    n_loc: int
    n_adj: int
    n_conn: int
    n_filler: int

    @property
    def n_special(self) -> int:
        return 3  # bos, eos, pad

    @property
    def total(self) -> int:
        return self.n_special + self.n_subj + self.n_verb + self.n_loc + self.n_adj + self.n_conn + self.n_filler


@dataclass
class DGPConfig:
    vocab: VocabSizes
    seq_len: int
    seed: int

    bigram_top_k: int  # number of favored verbs per subject (3)
    bigram_probs: tuple[float, ...]  # mass on the top-k favored verbs (e.g. 0.7, 0.2, 0.1)

    skip_trigram_n_pairs: int  # K distinct (LOC, ADJ) pairs (4)
    skip_trigram_max_skip: int  # window size for LOC lookback (8)

    induction_smoothing: float  # 0.1 means 0.9 on induced + 0.1 uniform

    # default-unigram category probabilities (must sum to 1.0)
    default_p_filler: float
    default_p_subj: float
    default_p_loc: float
    default_p_adj: float
    default_p_verb: float
    default_p_conn: float


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int

    d_model: int
    d_head: int  # full head dim, single head per layer
    d_mlp: int
    n_layers: int

    rope_base: float
    use_rmsnorm: bool


@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    betas: tuple[float, float]

    batch_size: int
    n_steps: int
    warmup_steps: int

    eval_every: int
    eval_n_batches: int
    checkpoint_steps: tuple[int, ...]  # steps at which to save

    seed: int
    num_data_workers: int


@dataclass
class RunConfig:
    dgp: DGPConfig
    model: ModelConfig
    train: TrainConfig
    out_dir: str


def default_dgp(seed: int = 0) -> DGPConfig:
    vocab = VocabSizes(n_subj=8, n_verb=12, n_loc=8, n_adj=8, n_conn=8, n_filler=16)
    return DGPConfig(
        vocab=vocab,
        seq_len=64,
        seed=seed,
        bigram_top_k=3,
        bigram_probs=(0.7, 0.2, 0.1),
        skip_trigram_n_pairs=24,
        skip_trigram_max_skip=12,
        induction_smoothing=0.1,
        default_p_filler=0.55,
        default_p_subj=0.10,
        default_p_loc=0.12,
        default_p_adj=0.10,
        default_p_verb=0.06,
        default_p_conn=0.07,
    )


def default_model(vocab_size: int, seq_len: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=64,
        d_head=32,
        d_mlp=128,
        n_layers=2,
        rope_base=10000.0,
        use_rmsnorm=True,
    )


def default_train(seed: int = 0) -> TrainConfig:
    return TrainConfig(
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        batch_size=128,
        n_steps=20000,
        warmup_steps=200,
        eval_every=1000,
        eval_n_batches=8,
        checkpoint_steps=(500, 1000, 2000, 5000, 10000, 20000),
        seed=seed,
        num_data_workers=0,
    )
