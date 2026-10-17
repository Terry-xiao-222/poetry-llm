import torch

def precompute_rope(head_dim, max_seq_len, base=10000.0):
    """预先算好 RoPE 用到的 cos 和 sin 表。"""
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, theta)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    """把向量"后一半取负、和前一半交换"。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """对 Q 和 K 施加旋转位置编码。"""
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


if __name__ == "__main__":
    B, n_heads, seq_len, head_dim = 2, 8, 16, 64
    q = torch.randn(B, n_heads, seq_len, head_dim)
    k = torch.randn(B, n_heads, seq_len, head_dim)

    cos, sin = precompute_rope(head_dim, max_seq_len=16)
    q_rot, k_rot = apply_rope(q, k, cos, sin)

    print("旋转后 Q 形状:", q_rot.shape)
    print("旋转前后模长一致吗:",
          torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5))
    