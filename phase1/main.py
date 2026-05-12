import argparse

from phase1.config import RunConfig, default_dgp, default_model, default_train
from phase1.train import train


def make_default_run(out_dir: str, seed: int = 0) -> RunConfig:
    dgp = default_dgp(seed=seed)
    model = default_model(vocab_size=dgp.vocab.total, seq_len=dgp.seq_len)
    tr = default_train(seed=seed)
    return RunConfig(dgp=dgp, model=model, train=tr, out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs/default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_data_workers", type=int, default=None)
    parser.add_argument("--init_std", type=float, default=None)
    parser.add_argument("--norm_init", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    args = parser.parse_args()
    cfg = make_default_run(out_dir=args.out, seed=args.seed)
    if args.n_steps is not None:
        cfg.train.n_steps = args.n_steps
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.eval_every is not None:
        cfg.train.eval_every = args.eval_every
    if args.num_data_workers is not None:
        cfg.train.num_data_workers = args.num_data_workers
    if args.init_std is not None:
        cfg.model.init_std = args.init_std
    if args.norm_init is not None:
        cfg.model.norm_init = args.norm_init
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    train(cfg)


if __name__ == "__main__":
    main()
