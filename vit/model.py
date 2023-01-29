import torch
from torch import nn

# Paper: https://arxiv.org/pdf/2010.11929.pdf

H, W, P, D = 224, 224, 16, 768
N = int(H*W/P**2)

class PatchEmbedding(nn.Module):
  # Input: 2D image
  # Output: 1D embedding vector
  def __init__(self, C=3, P=16, D=768):
    super().__init__()

    # Turn img to patches
    self.patch = nn.Conv2d(
      in_channels=C,
      out_channels=D,
      kernel_size=P,
      stride=P,
      padding=0
    )

    # Flatten patch feature maps into a single dimension
    self.flatten = nn.Flatten(start_dim=2, end_dim=3)

  def forward(self, x):
    x = self.patch(x)
    x = self.flatten(x)
    return x.permute(0, 2, 1)

class MSA(nn.Module):
  def __init__(self, D, dropout=0, n_heads=12):
    super().__init__()

    self.ln = nn.LayerNorm(normalized_shape=D)
    self.msa = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)

  def forward(self, x):
    x = self.ln(x)
    attn_out, _ = self.msa(x, x, x, need_weights=False)
    return attn_out

class MLP(nn.Module):
  def __init__(self, D, dropout=0.1, mlp_size=3072):
    super().__init__()

    self.ln = nn.LayerNorm(normalized_shape=D)
    self.mlp = nn.Sequential(
      nn.Linear(in_features=D, out_features=mlp_size),
      nn.GELU(),
      nn.Dropout(p=dropout),
      nn.Linear(in_features=mlp_size, out_features=D),
      nn.Dropout(p=dropout)
    )

  def forward(self, x):
    x = self.ln(x)
    x = self.mlp(x)
    return x

class TransformerEncoder(nn.Module):
  def __init__(self, D):
    super().__init__()

    self.msa = MSA(D)
    self.mlp = MLP(D)

  def forward(self, x):
    # We add residual connection to both block
    x = self.msa(x) + x
    x = self.mlp(x) + x
    return x

class ViT(nn.Module):
  def __init__(self, D, num_classes, n_tf_layers=12):
    super().__init__()

    self.encoder = nn.Sequential(
      *[TransformerEncoder(D) for _ in range(n_tf_layers)]
    )

    self.classifier_head = nn.Sequential(
      nn.LayerNorm(normalized_shape=D),
      nn.Linear(in_features=D, out_features=num_classes)
    )

  # This is the input for the transformer encoder
  def token_embeddings(self, x):
    # Turn img to embedded patches
    em = PatchEmbedding()
    if x.dim() == 3: x_p = em(x.unsqueeze(0))
    else: x_p = em(x)

    # Prepend class token to sequence of patches
    batch_size = x_p.shape[0]
    x_class = nn.Parameter(torch.randn(batch_size, 1, D))
    x_p = torch.cat((x_class, x_p), dim=1)

    # Add position embedding to patches
    e_pos = nn.Parameter(torch.randn((1, N+1, D)))
    z_0 = x_p + e_pos
    return z_0

  def forward(self, x):
    x = self.token_embeddings(x)
    x = self.encoder(x)
    x = self.classifier_head(x[:,0])
    return x
