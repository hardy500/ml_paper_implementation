import torchvision
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pretrained_w = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_model = torchvision.models.vit_b_16(weights=pretrained_w).to(device=device)

for param in pretrained_model.parameters():
  param.requires_grad = False

torch.manual_seed(42)
pretrained_model.heads = nn.Linear(in_features=768, out_features=3)

auto_transform = pretrained_w.transforms()