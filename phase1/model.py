import math

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

from phase1.config import ModelConfig, NormType


def precompute_rope(
    seq_len: int, head_dim: int, base: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE; got {head_dim}"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: Float[Tensor, "B S D"], cos: Float[Tensor, "S half"], sin: Float[Tensor, "S half"]
) -> Float[Tensor, "B S D"]:
    """Split-in-half RoPE."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, :, :].to(x.dtype)
    sin = sin[None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def make_causal_mask(seq_len: int, device: torch.device) -> Bool[Tensor, "S S"]:
    """True at positions to be zeroed (key index > query index)."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class BilinearAttention(nn.Module):
    """Single-head bilinear attention.

    A1 = q1 k1^T / sqrt(d_head)
    A2 = q2 k2^T / sqrt(d_head)
    A  = A1 ⊙ A2, causal-masked (zeroed where key > query). No softmax.
    out = A V W_O
    """

    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.q1_proj = nn.Linear(d_model, d_head, bias=False)
        self.q2_proj = nn.Linear(d_model, d_head, bias=False)
        self.k1_proj = nn.Linear(d_model, d_head, bias=False)
        self.k2_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.o_proj = nn.Linear(d_head, d_model, bias=False)

    def forward(
        self,
        x: Float[Tensor, "B S D"],
        cos: Float[Tensor, "S half"],
        sin: Float[Tensor, "S half"],
        causal_mask: Bool[Tensor, "S S"],
    ) -> Float[Tensor, "B S D"]:
        q1 = apply_rope(self.q1_proj(x), cos, sin)
        q2 = apply_rope(self.q2_proj(x), cos, sin)
        k1 = apply_rope(self.k1_proj(x), cos, sin)
        k2 = apply_rope(self.k2_proj(x), cos, sin)
        v = self.v_proj(x)
        scale = 1.0 / math.sqrt(self.d_head)
        a1 = einsum(q1, k1, "b s d, b t d -> b s t") * scale
        a2 = einsum(q2, k2, "b s d, b t d -> b s t") * scale
        a = a1 * a2
        a = a.masked_fill(causal_mask, 0.0)
        out = einsum(a, v, "b s t, b t d -> b s d")
        return self.o_proj(out)


class BilinearMLP(nn.Module):
    """Pearce-style bilinear MLP: (W_m x) ⊙ (W_n x) projected back to d_model."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.w_m = nn.Linear(d_model, d_mlp, bias=False)
        self.w_n = nn.Linear(d_model, d_mlp, bias=False)
        self.w_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: Float[Tensor, "B S D"]) -> Float[Tensor, "B S D"]:
        return self.w_proj(self.w_m(x) * self.w_n(x))


class RMSNorm(nn.Module):
    """Per-token RMS normalization (non-polynomial in inputs; included for the
    interpretability-trade-off baseline only — VPD-aligned configs avoid this)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LearnableScalar(nn.Module):
    """y = s · x with a single learnable scalar s. Linear in both x and s."""

    def __init__(self, init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init))

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        return x * self.scale


class LearnableChannelScale(nn.Module):
    """y = diag(s) · x, learnable per-channel scale of length d_model.
    Linear in both x and s."""

    def __init__(self, d_model: int, init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((d_model,), init))

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        return x * self.scale


def make_norm(norm_type: NormType, d_model: int, init: float) -> nn.Module:
    match norm_type:
        case "none":
            return nn.Identity()
        case "scalar":
            return LearnableScalar(init=init)
        case "channel":
            return LearnableChannelScale(d_model, init=init)
        case "rmsnorm":
            return RMSNorm()


class BilinearBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = BilinearAttention(cfg.d_model, cfg.d_head)
        self.mlp = BilinearMLP(cfg.d_model, cfg.d_mlp)
        self.norm_attn = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_init)
        self.norm_mlp = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_init)

    def forward(
        self,
        x: Float[Tensor, "B S D"],
        cos: Float[Tensor, "S half"],
        sin: Float[Tensor, "S half"],
        causal_mask: Bool[Tensor, "S S"],
    ) -> Float[Tensor, "B S D"]:
        x = x + self.attn(self.norm_attn(x), cos, sin, causal_mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class BilinearTransformer(nn.Module):
    """Decoder-only fully-bilinear transformer.

    No layernorm, no softmax, no biases. Each weight matrix is its own nn.Linear so
    later VPD analysis can address them by path.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([BilinearBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg.norm_type, cfg.d_model, cfg.norm_init)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()
        cos, sin = precompute_rope(cfg.seq_len, cfg.d_head, cfg.rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer(
            "causal_mask", make_causal_mask(cfg.seq_len, torch.device("cpu")), persistent=False
        )

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, tokens: Int[Tensor, "B S"]) -> Float[Tensor, "B S V"]:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, self.causal_mask)
        return self.unembed(self.final_norm(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
