import torch.nn as nn
import torch

binary_cross_entropy_loss = nn.BCELoss()
l1 = torch.tensor([0.])
l2 = torch.tensor([1.0])
print(binary_cross_entropy_loss(l1, l2))
