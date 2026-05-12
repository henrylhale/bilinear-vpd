import torch

from phase1.config import default_dgp, default_model
from phase1.model import (
    BilinearTransformer,
    LearnableChannelScale,
    apply_rope,
    make_causal_mask,
    precompute_rope,
)


def test_forward_shape_and_finite():
    dgp_cfg = default_dgp()
    cfg = default_model(vocab_size=dgp_cfg.vocab.total, seq_len=dgp_cfg.seq_len)
    model = BilinearTransformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len))
    logits = model(tokens)
    assert logits.shape == (4, cfg.seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_causal_attention():
    """Changing tokens at position t must not affect logits at positions < t."""
    cfg = default_model(vocab_size=63, seq_len=16)
    model = BilinearTransformer(cfg)
    model.eval()
    torch.manual_seed(0)
    tokens_a = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    tokens_b = tokens_a.clone()
    flip_pos = cfg.seq_len - 1
    tokens_b[0, flip_pos] = (tokens_b[0, flip_pos] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_a = model(tokens_a)
        logits_b = model(tokens_b)
    assert torch.allclose(logits_a[0, :flip_pos], logits_b[0, :flip_pos], atol=1e-6)


def test_no_layernorm_or_rmsnorm():
    """The network must be parameter-polynomial — no per-token RMS or layer norms."""
    import torch.nn as nn
    cfg = default_model(vocab_size=63, seq_len=16)
    model = BilinearTransformer(cfg)
    for m in model.modules():
        assert not isinstance(m, nn.LayerNorm)
        assert not isinstance(m, nn.RMSNorm)


def test_no_biases():
    cfg = default_model(vocab_size=63, seq_len=16)
    model = BilinearTransformer(cfg)
    for n, p in model.named_parameters():
        assert "bias" not in n, f"unexpected bias parameter: {n}"


def test_named_module_paths_for_vpd():
    """Each weight should be reachable by stable submodule path; sanity-check a few."""
    cfg = default_model(vocab_size=63, seq_len=16)
    model = BilinearTransformer(cfg)
    expected = [
        "embed",
        "unembed",
        "blocks.0.attn.q1_proj",
        "blocks.0.attn.q2_proj",
        "blocks.0.attn.k1_proj",
        "blocks.0.attn.k2_proj",
        "blocks.0.attn.v_proj",
        "blocks.0.attn.o_proj",
        "blocks.0.mlp.w_m",
        "blocks.0.mlp.w_n",
        "blocks.0.mlp.w_proj",
        "blocks.1.attn.q1_proj",
        "blocks.1.mlp.w_proj",
    ]
    actual = {n for n, _ in model.named_modules()}
    for path in expected:
        assert path in actual, f"missing module path {path}"


def test_rope_cos_sin_shapes():
    cos, sin = precompute_rope(seq_len=16, head_dim=32, base=10000.0, device=torch.device("cpu"))
    assert cos.shape == (16, 16)
    assert sin.shape == (16, 16)


def test_apply_rope_preserves_shape_and_norm():
    torch.manual_seed(0)
    cos, sin = precompute_rope(seq_len=8, head_dim=16, base=10000.0, device=torch.device("cpu"))
    x = torch.randn(2, 8, 16)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    # RoPE is a rotation -> per-token L2 norm preserved.
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_causal_mask_correctness():
    m = make_causal_mask(5, torch.device("cpu"))
    assert m.shape == (5, 5)
    expected = torch.tensor(
        [
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ]
    )
    assert torch.equal(m, expected)


def test_channel_scale_is_linear_in_input():
    """LearnableChannelScale must be linear in x (key VPD invariant): doubling x
    must double the output. This is the property that makes the network
    parameter-polynomial all the way through."""
    norm = LearnableChannelScale(d_model=8, init=2.0)
    x = torch.randn(2, 4, 8)
    assert torch.allclose(norm(2 * x), 2 * norm(x), atol=1e-6)


def test_channel_scale_is_linear_in_parameter():
    """And linear in the scale parameter too — the (params, input) bilinear property."""
    d = 8
    norm = LearnableChannelScale(d_model=d, init=1.0)
    x = torch.randn(2, 4, d)
    y1 = norm(x)
    with torch.no_grad():
        norm.scale.mul_(3.0)
    y3 = norm(x)
    assert torch.allclose(y3, 3 * y1, atol=1e-6)


def test_loss_decreases_on_smoke_batch():
    """A few hundred steps of training on a single fixed batch should reduce loss."""
    torch.manual_seed(0)
    dgp_cfg = default_dgp()
    cfg = default_model(vocab_size=dgp_cfg.vocab.total, seq_len=dgp_cfg.seq_len)
    model = BilinearTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tokens = torch.randint(0, cfg.vocab_size, (8, cfg.seq_len))
    initial_loss = None
    final_loss = None
    for step in range(200):
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        if step == 0:
            initial_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = loss.item()
    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.5, f"loss did not drop enough: {initial_loss} -> {final_loss}"
