import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms_inv * self.weight


if __name__ == "__main__":
    norm = RMSNorm(dim=512)
    x = torch.randn(2, 10, 512)
    out = norm(x)

    print("输出形状:", out.shape)

    rms_after = out.pow(2).mean(dim=-1).sqrt().mean().item()
    print(f"归一化后平均 RMS: {rms_after:.6f}")

    n_params = sum(p.numel() for p in norm.parameters())
    print(f"参数量: {n_params}  (LayerNorm 会是 {n_params * 2})")
    