import json
from pathlib import Path

import pytest

from phase1.config import RunConfig, default_dgp, default_model, default_train
from phase1.train import train


@pytest.fixture
def tiny_run(tmp_path: Path) -> RunConfig:
    dgp = default_dgp(seed=0)
    model = default_model(vocab_size=dgp.vocab.total, seq_len=dgp.seq_len)
    tr = default_train(seed=0)
    tr.n_steps = 200
    tr.warmup_steps = 20
    tr.eval_every = 100
    tr.eval_n_batches = 2
    tr.batch_size = 32
    tr.checkpoint_steps = (100,)
    return RunConfig(dgp=dgp, model=model, train=tr, out_dir=str(tmp_path / "run"))


def test_smoke_training_run_drops_loss(tiny_run: RunConfig):
    train(tiny_run)
    log_path = Path(tiny_run.out_dir) / "log.jsonl"
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(lines) >= 2  # eval at step 0, 100, 200
    first = lines[0]
    last = lines[-1]
    assert last["eval_overall_loss"] < first["eval_overall_loss"], (
        f"loss did not drop: {first['eval_overall_loss']:.4f} -> {last['eval_overall_loss']:.4f}"
    )
    assert (Path(tiny_run.out_dir) / "model_final.pt").exists()
    assert (Path(tiny_run.out_dir) / "config.json").exists()
    assert (Path(tiny_run.out_dir) / "model_step_100.pt").exists()
