import torch.nn as nn

class Basic(nn.Module):
  def __init__(self):
    super(Basic, self).__init__()
    self.c1 = nn.Conv1d(1, 32, 160)

  def forward(self, X):
    return self.c1(X)