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

class SwiGLU(nn.Module):
  def __init__(self, d_model : int, d_ff = None, device=None, dtype=None):
    super().__init__()
    if not d_ff:
      d_ff = (((d_model * 8) // 3) // 64) * 64

    self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
    self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    ff = self.w1(x)
    gate = self.w3(x)
    silu = torch.sigmoid(ff) * ff

    return self.w2(silu * gate)

class RotaryPositionalEmbedding(nn.Module):
  cos_buffer : torch.Tensor
  sin_buffer : torch.Tensor

  def __init__(self, theta : float, d_k : int, max_seq_len : int, device=None):
    super().__init__()

    positions = torch.arange(0, max_seq_len, device=device)
    ks = torch.repeat_interleave(torch.arange(1, (d_k // 2) + 1, device=device), 2)
    thetas = einsum(positions, (torch.pow(theta, -((2 * ks) - 2) / d_k)), "i, d -> i d")

    cos = torch.cos(thetas)
    sin = torch.sin(thetas)

    self.register_buffer("cos_buffer", cos, persistent=False)
    self.register_buffer("sin_buffer", sin, persistent=False)


  def forward(self, x : torch.Tensor, token_positions : torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    cos = self.cos_buffer[token_positions]
    sin = self.sin_buffer[token_positions]

    og = -x[..., 1::2]
    other = x[..., ::2]
    swapped = rearrange([og, other], "stack ... d -> ... (d stack)")

    return (x * cos) + (swapped * sin)

def softmax(x : torch.Tensor, i) -> torch.Tensor:
  m, _ = torch.max(x, i, keepdim=True)
  scaled = x - m
  numer = torch.exp(scaled)
  denom = torch.sum(numer, i, keepdim=True)
  return numer / denom

def scaled_dot_product_attention(Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, mask=None):
  inner = einsum(Q, K, "... q_len d_k, ... k_len d_k -> ... q_len k_len")
  inner /= math.sqrt(Q.shape[-1])

  if mask is not None:
    inner.masked_fill_(~mask, value=-float('inf'))

  return einsum(softmax(inner, -1), V, "... q_len k_len, ... k_len d_v -> ... q_len d_v")