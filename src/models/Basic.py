import torch.nn as nn

class Basic(nn.Module):
  def __init__(self, n_classes):
    super(Basic, self).__init__()
    self.c1 = nn.Conv1d(1, n_classes, 32)
    self.log_softmax = nn.LogSoftmax(dim = 1)

  def forward(self, X):
    a = self.c1(X)
    return self.log_softmax(a)