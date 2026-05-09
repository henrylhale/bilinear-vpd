# bilinear-vpd

Phase 1 of an interpretability research project: a 2-layer fully-bilinear (no-softmax / no-elementwise-nonlinearity) decoder transformer trained on a synthetic data-generating process with three known computational primitives — bigram, skip-trigram, and induction. A trained model from this phase becomes the target for adversarial Parameter Decomposition (VPD) in Phase 2.

The project spec lives in [`SPEC.md`](SPEC.md). Phase-1 code lives in [`phase1/`](phase1/). Everything outside `phase1/` is the upstream `goodfire-ai/param-decomp` repository, kept as scaffolding for Phase 2 (where the VPD machinery in `param_decomp/` will be applied to our trained model). The upstream README and CLAUDE.md describe that scaffolding; this README covers Phase 1 only.

## What this phase produces

- A synthetic DGP (`phase1/data.py`) that emits sequences over a ~63-token vocabulary and per-position annotations marking which primitive (if any) determines the next token.
- A 2-layer bilinear transformer (`phase1/model.py`):
  - Bilinear attention: two QK pairs whose patterns are combined via elementwise product (no softmax), causal-masked, with `1/√d_head` scaling.
  - Bilinear MLP: `(W_m x) ⊙ (W_n x) W_proj` (Pearce et al.).
  - RoPE on Q and K. No biases, no LayerNorm/RMSNorm.
  - Each weight matrix is its own `nn.Linear` so VPD can address them by submodule path.
- A training loop (`phase1/train.py`) and per-primitive evaluator (`phase1/eval.py`) that report KL/CE/top-1 on positions where each primitive fires.

## Setup

```bash
~/miniconda3/bin/conda create -y -n bilinear-vpd python=3.11
~/miniconda3/envs/bilinear-vpd/bin/pip install \
    torch --index-url https://download.pytorch.org/whl/cpu
~/miniconda3/envs/bilinear-vpd/bin/pip install einops jaxtyping numpy pytest tqdm
```

The Phase-1 code does not depend on the param-decomp package — only `torch`, `numpy`, `einops`, `jaxtyping`.

## Run

From the repo root:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 \
    ~/miniconda3/envs/bilinear-vpd/bin/python -m phase1.main \
    --out runs/v5 --n_steps 25000 --batch_size 512 --lr 1e-3 \
    --eval_every 1000 --num_data_workers 32
```

These are the hyperparameters of the trained checkpoint shipped in `runs/v5/` — they hit all per-primitive KL targets on a single RTX 3090 in about 6 minutes. On CPU only, drop `num_data_workers` to 0 and reduce `batch_size`/`n_steps`.

Outputs land in the chosen `--out` directory:

- `config.json` — full run config (DGP seed, model dims, training hyperparams)
- `log.jsonl` — one record per eval, with overall loss + per-primitive KL/top-1
- `model_step_*.pt`, `model_final.pt` — checkpoints

`OMP_NUM_THREADS=1` is intentional on small machines — for this model size, multi-threaded BLAS contends with itself and is slower than single-threaded.

## Tests

```bash
PYTHONPATH=. ~/miniconda3/envs/bilinear-vpd/bin/python -m pytest phase1/tests -v
```

The tests exercise:

- DGP correctness: bigram empirical distribution vs. the rule, skip-trigram determinism, induction uniqueness, primitive firing rates roughly in [3%, 20%].
- Model: forward shape + finiteness, causality (changing token at position `t` leaves logits at `<t` unchanged), no LayerNorm / no biases, named submodule paths required by VPD.
- Training: a few-hundred-step smoke run reduces eval loss measurably, checkpoints are written.

## Reading the eval output

Each eval log line contains:

| field             | meaning                                                              |
| ----------------- | -------------------------------------------------------------------- |
| `eval_overall_loss` | Mean cross-entropy over all next-token predictions in eval batches |
| `kl_<primitive>`  | Mean `KL(p_true ∥ p_model)` in nats over positions where the primitive fires |
| `h_<primitive>`   | Mean entropy of the ground-truth distribution `H(p_true)`            |
| `top1_<primitive>` | Fraction of positions where `argmax(model)` matches `argmax(p_true)` |
| `n_<primitive>`   | Number of eval positions where the primitive fired                   |

Phase-1 success criterion (from the spec): `kl_bigram ≤ 0.1`, `kl_skip ≤ 0.1`, `kl_induction ≤ 0.3`. The skip-trigram primitive is deterministic so `H_skip = 0` and `kl_skip = ce_skip = -log p_model[true]`. For bigram and induction, `H_*` is positive — the floor `kl = 0` corresponds to the model exactly matching the ground-truth conditional.

The shipped `runs/v5/model_final.pt` checkpoint hits:

| primitive   | KL (nats) | top-1 | target KL |
| ----------- | --------- | ----- | --------- |
| bigram      | 0.002     | 1.000 | ≤ 0.1     |
| skip-trigram | 0.031    | 0.993 | ≤ 0.1     |
| induction   | 0.269     | 0.951 | ≤ 0.3     |

Induction goes through a sharp phase change around step 8k–10k where KL drops from ~2.8 to ~1.0 over a few hundred steps; before that it is stuck at the entropy of "uniform over content tokens".

## Key design choices

These were decided during the planning conversation (also recorded at the bottom of `SPEC.md`):

- **2 layers** instead of 1 (induction needs composition: previous-token information in layer 1 feeding a copying head in layer 2).
- **Smoothed induction**: ground truth at induction positions is `0.9 · δ_induced + 0.1 · uniform_over_content_tokens` rather than a hard delta — avoids pathological infinite-KL during training.
- **`1/√d_head` scaling** on each raw QK product before the elementwise product, for stability without softmax.
- **No biases anywhere** — Q/K/V/O, both MLP arms, output projection, unembed.
- **RMSNorm before each sublayer + before unembed** (no learnable affine — the norm is the only non-bilinear operation in the network). Without it the induction circuit does not form within 50k steps; with it the circuit emerges around step 8k. Toggle via `ModelConfig.use_rmsnorm`.
- **All weight matrices are separate `nn.Linear` modules** at predictable paths (`blocks.{i}.attn.{q1,q2,k1,k2,v,o}_proj`, `blocks.{i}.mlp.{w_m,w_n,w_proj}`) — required by `param_decomp.models.component_model.ComponentModel`.

## Phase-2 hand-off

A trained checkpoint + its `config.json` is enough to reconstruct everything for VPD:

```python
import json, torch
from phase1.config import RunConfig, DGPConfig, ModelConfig, TrainConfig, VocabSizes
from phase1.data import DGP
from phase1.model import BilinearTransformer

with open("runs/v1/config.json") as f:
    raw = json.load(f)
# (deserialize each nested dataclass; trivial — every field is a primitive or tuple)

dgp = DGP(cfg.dgp)            # rule tables fully determined by cfg.dgp.seed
model = BilinearTransformer(cfg.model)
model.load_state_dict(torch.load("runs/v1/model_final.pt"))
```

The DGP's `subj_verb_dist` and `skip_rules` are accessible as `dgp.rules.subj_verb_dist` and `dgp.rules.skip_rules` — Phase 2 will use these as ground truth when asking "is this circuit implementing rule X?".
