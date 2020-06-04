import torch.nn as nn
import torch.nn.functional as F

# https://fomoro.com/research/article/receptive-field-calculator#400,200,1,VALID;3,1,2,SAME;3,1,1,SAME;3,2,2,SAME;3,1,1,SAME;5,1,2,SAME;5,1,1,SAME;5,1,2,SAME;5,1,1,SAME


class DilatedResNet(nn.Module):

  def __init__(self, n_classes):
    super(DilatedResNet, self).__init__()
    self.a1 = nn.Conv2d(1, 64, kernel_size=1, stride=1)

    # Resnet block 1
    self.c1 = nn.Conv2d(
        64, 64, kernel_size=3, stride=1, padding=(1, 2), dilation=(1, 2))
    self.bn1 = nn.BatchNorm2d(64)
    self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)

    self.mp1 = nn.MaxPool2d((2, 1), stride=2)
    self.a2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)

    # Resnet block 2
    self.c3 = nn.Conv2d(
        128, 128, kernel_size=3, stride=1, padding=(1, 2), dilation=(1, 2))
    self.bn3 = nn.BatchNorm2d(128)
    self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(128)

    self.mp2 = nn.MaxPool2d((2, 1))

    # Resnet block 3
    self.c5 = nn.Conv2d(
        128, 128, kernel_size=5, stride=1, padding=(2, 4), dilation=(1, 2))
    self.bn5 = nn.BatchNorm2d(128)
    self.c6 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
    self.bn6 = nn.BatchNorm2d(128)

    self.mp3 = nn.MaxPool2d((2, 1))
    self.a3 = nn.Conv2d(128, 256, kernel_size=1, stride=1)

    # Resnet block 4
    self.c7 = nn.Conv2d(
        256, 256, kernel_size=5, stride=1, padding=(2, 4), dilation=(1, 2))
    self.bn7 = nn.BatchNorm2d(256)
    self.c8 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2)
    self.bn8 = nn.BatchNorm2d(256)

    self.mp4 = nn.MaxPool2d((2, 1))

    self.c9 = nn.Conv1d(256 * 4, n_classes, kernel_size=1, stride=1)

  def forward(self, mfcc):
    out = self.a1(mfcc)
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
    out = self.a3(out)
    out = F.relu(out)

    # Resnet block 4
    o7 = self.c7(out)
    b7 = self.bn7(o7)
    a4 = F.relu(b7)
    o8 = self.c8(a4)
    b8 = self.bn8(o8)
    c4 = b8 + out
    out = F.relu(c4)

    # Max pool
    out = self.mp3(out)

    # Flatten
    N, C, H, T = out.shape
    out = out.view(N, C * H, T)

    # Final 1D convolution
    out = self.c9(out)
    out = F.log_softmax(out, dim=1)

    return out

  def forward_shape(self, lengths):
    out_lengths = lengths // 2
    return out_lengths