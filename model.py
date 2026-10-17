import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from RMSNorm import RMSNorm
from RoPE import precompute_rope, apply_rope
from SwiGLU import SwiGLU

class Attention(nn.Module):
    """带 RoPE 的 GQA 注意力。"""
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """一个 Transformer 层 = 注意力子层 + FFN 子层，各自带 Pre-Norm 和残差。"""
    def __init__(self, dim, n_heads, n_kv_heads, ffn_hidden):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=8, n_heads=8,
                 n_kv_heads=2, ffn_hidden=1536, max_seq_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, ffn_hidden)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Transformer 默认初始化会让 tied embedding 产生过大的初始 logits。
        # 先初始化，再绑定输入/输出权重，使初始交叉熵接近 ln(vocab_size)。
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.wo.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn.w_down.weight, mean=0.0, std=residual_std)
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(dim // n_heads, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        cos, sin = self.cos[:L], self.sin[:L]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)


# ====== 验证：前向 + 参数量 ======
if __name__ == "__main__":
    model = MiniLLM(vocab_size=32000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.1f}M")

    ids = torch.randint(0, 32000, (2, 128))
    logits = model(ids)
    print(f"输入: {ids.shape} → 输出 logits: {logits.shape}")
