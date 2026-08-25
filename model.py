import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

class CausalMultiheadAttention(nn.Module):
    freqs_cis: torch.Tensor

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O._is_residual_proj: bool = True  # type: ignore

        self.max_seq_len = cfg.max_seq_len
        self.rope_base = cfg.rope_base

        j = torch.arange(0, self.d_head // 2, dtype=torch.float32)
        inv_freq = self.rope_base ** (-2 * j / self.d_head)
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)[None, None, :, :]

        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model, (
            f"d_model does not match expected d_model: {d_model} != {self.d_model}"
        )
        Q = self.W_Q(x)  # (batch_size, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # reshape to (batch_size, seq_len, n_heads, d_head)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        Q = self.apply_rope(Q)
        K = self.apply_rope(K)

        attn_before_proj = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        # (batch_size, n_heads, seq_len, d_head)

        # reshape to (batch_size, seq_len, n_heads, d_head)
        attn_before_proj = attn_before_proj.transpose(1, 2)

        # reshape to (batch_size, seq_len, d_model)
        attn_before_proj = attn_before_proj.reshape(batch_size, seq_len, self.d_model)

        # project to d_model
        return self.W_O(attn_before_proj)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        batch_size, n_heads, seq_len, d_head = x.size()
        assert d_head == self.d_head, (
            f"d_head does not match expected d_head: {d_head} != {self.d_head}"
        )
        assert seq_len <= self.max_seq_len, (
            f"seq_len is greater than max_seq_len: {seq_len} > {self.max_seq_len}"
        )
        x = x.reshape(batch_size, n_heads, seq_len, d_head // 2, 2)
        x = torch.view_as_complex(x.float()) * self.freqs_cis[..., :seq_len, :]
        x = torch.view_as_real(x).reshape(batch_size, n_heads, seq_len, d_head)
        return x.to(orig_dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_hidden, bias=False)
        self.w_down = nn.Linear(cfg.d_hidden, cfg.d_model, bias=False)
        self.w_down._is_residual_proj: bool = True  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalMultiheadAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg=self.cfg,
                )
                for _ in range(self.cfg.n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            input_dim = module.in_features
            std = input_dim ** -0.5
            if getattr(module, "_is_residual_proj", False):
                std *= (2 * self.cfg.n_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.d_model**-0.5)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # idx: (B, T) int64
        x = self.tok_emb(idx)  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)  # (B, T, vocab_size) — логиты
