import torch.nn as nn
import torch.nn.functional as F

class Basic(nn.Module):
  def __init__(self, n_classes):
    super(Basic, self).__init__()
    self.c1 = nn.Conv1d(1, 128, 32)
    self.c2 = nn.Conv1d(128, n_classes, 64)
    self.log_softmax = nn.LogSoftmax(dim = 1)

  def forward(self, X):
    a = self.c1(X)
    b = F.relu(a)
    c = self.c2(b)
    return self.log_softmax(c, dim=1)

  def forward_shape(self, lengths):
    return lengths - 32 - 64 + 1 + 1