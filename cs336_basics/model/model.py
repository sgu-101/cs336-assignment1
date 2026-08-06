from einops import rearrange, reduce, repeat, einsum
import numpy as np
import torch
import torch.nn as nn
import math

class Linear(nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype=None):
    super().__init__()

    self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
    std = math.sqrt(2 / (in_features + out_features))
    nn.init.trunc_normal_(self.weight, mean = 0, std=std, a = -3.0 * std, b = 3.0 * std)

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")

class Embedding(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
    super().__init__()

    self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
    std = 1
    nn.init.trunc_normal_(self.weight, mean = 0, std=std, a = -3.0, b = 3.0)

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    return self.weight[x]

class RMSNorm(nn.Module):
  def __init__(self, d_model : int, eps : float = 1e-5, device=None, dtype=None):
    super().__init__()

    self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    self.eps = eps

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)

    means = reduce(torch.square(x), "... i -> ... 1", "mean")
    RMS = torch.sqrt(repeat(means, "... 1 -> ... i", i=x.shape[-1]) + self.eps)
    result = torch.mul(torch.div(x, RMS), self.gain)

    return result.to(in_dtype)
    