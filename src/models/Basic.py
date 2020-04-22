import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MFCC


class Basic(nn.Module):

  def __init__(self, n_classes):
    super(Basic, self).__init__()
    self.mfcc = MFCC()
    # Height after: 40 - 9 + 1 = 32
    self.c1 = nn.Conv2d(1, 128, kernel_size=9, stride=1)
    # Height after: 32 - 17 + 1 = 16
    self.c2 = nn.Conv2d(128, 256, kernel_size=17, stride=1)
    self.c3 = nn.Conv1d(256 * 16, n_classes, kernel_size=1, stride=1)

  def forward(self, X):
    out = self.mfcc(X)
    out = F.relu(self.c1(out))
    out = F.relu(self.c2(out))
    N, C, H, T = out.shape
    out = out.view(N, C * H, T)
    out = F.log_softmax(self.c3(out), dim=1)
    return out

  def forward_shape(self, lengths):
    return lengths // (400 / 2) + 1 - 8 - 16