import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.skip_conv = None
    if stride != 1 or inplanes != planes:
      self.skip_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
      self.skip_bn = nn.BatchNorm2d(planes)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out, inplace=True)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.skip_conv is not None:
      residual = self.skip_conv(x)
      residual = self.skip_bn(residual)

    out += residual
    out = F.relu(out, inplace=True)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1):
    super(Bottleneck, self).__init__()

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)

    self.skip_conv = None
    if stride != 1 or inplanes != planes * 4:
      self.skip_conv = nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False)
      self.skip_bn = nn.BatchNorm2d(planes * 4)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out, inplace=True)

    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out, inplace=True)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.skip_conv is not None:
      out = self.skip_conv(x)
      residual = self.skip_bn(out)

    out += residual
    out = F.relu(out, inplace=True)
    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    strides = [1] * layers[0] + \
              [2] + [1] * (layers[1] - 1) + \
              [2] + [1] * (layers[2] - 1) + \
              [2] + [1] * (layers[3] - 1)
    out_channels = [64] * layers[0] + \
                   [128] * layers[1] + \
                   [256] * layers[2] + \
                   [512] * layers[3]

    self.layers = nn.ModuleList()
    last_c = 64
    for channel, stride in zip(out_channels, strides):
      self.layers.append(block(last_c, channel, stride))
      last_c = channel * block.expansion

    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.bn1(x), inplace=True)
    x = self.maxpool(x)

    for layer in self.layers:
      x = layer(x)

    x = x.mean(3).mean(2)
    x = self.fc(x)

    return x


def resnet18(num_classes=1000):
  return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=1000):
  return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=1000):
  return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


if __name__ == '__main__':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)


  net = resnet18()

  print('total num of parameters: %.5f' %
        (sum(p[1].data.nelement() for p in net.named_parameters()) / 1024 / 1024))

  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 224, 224))
  print(y.size())
