import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MFCC


class ResNet(nn.Module):

  def __init__(self, n_classes):
    super(ResNet, self).__init__()
    self.mfcc = MFCC(n_mfcc=64)

    self.a1 = nn.Conv2d(1, 64, kernel_size=1, stride=1)

    self.c1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.mp1 = nn.MaxPool2d(2)
    # Height after: (64 - 3 + 3) / 2 = 31

    self.a2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)

    self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(128)
    self.mp2 = nn.MaxPool2d(2)
    # Height after: (31 - 4 + 3) / 2 = 14

    self.c5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(128)
    self.c6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(128)
    self.mp3 = nn.MaxPool2d(2)
    # Height after: (14 - 5 + 3) / 2 = 5

    self.c7 = nn.Conv1d(128 * 8, n_classes, kernel_size=1, stride=1)

  def forward(self, X):
    # Make spectogram
    out = self.mfcc(X)
    out = self.a1(out)
    out = F.relu(out)

    # Resnet block 1
    o1 = self.c1(out)
    b1 = self.bn1(o1)
    a1 = F.relu(b1)
    o2 = self.c2(a1)
    b2 = self.bn2(o2)
    c1 = b2 + out
    out = F.relu(c1)

    # Max pool
    out = self.mp1(out)
    out = self.a2(out)
    out = F.relu(out)

    # Resnet block 2
    o3 = self.c3(out)
    b3 = self.bn3(o3)
    a2 = F.relu(b3)
    o4 = self.c4(a2)
    b4 = self.bn4(o4)
    c2 = b4 + out
    out = F.relu(c2)

    # Max pool
    out = self.mp2(out)

    # Resnet block 3
    o5 = self.c5(out)
    b5 = self.bn5(o5)
    a3 = F.relu(b5)
    o6 = self.c6(a3)
    b6 = self.bn6(o6)
    c3 = b6 + out
    out = F.relu(c3)

    # Max pool
    out = self.mp3(out)

    # Flatten
    N, C, H, T = out.shape
    out = out.view(N, C * H, T)

    # Final 1D convolution
    out = self.c7(out)
    out = F.log_softmax(out, dim=1)

    return out

  def forward_shape(self, lengths):
    out_lengths = lengths // (400 // 2)
    out_lengths = out_lengths // 2  # ResNet 2
    out_lengths = out_lengths // 2  # ResNet 2
    out_lengths = out_lengths // 2  # ResNet 3
    return out_lengths