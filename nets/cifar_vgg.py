import math

import torch
import torch.nn as nn


class standard_block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(standard_block, self).__init__()
    self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.conv2d(x)
    out = self.bn(out)
    out = self.relu(out)
    return out


class VGG(nn.Module):
  def __init__(self, conv_config, fc_config, num_classes):
    super(VGG, self).__init__()
    layers = []
    in_channels = 3

    for v in conv_config:
      if v == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      else:
        layers += [standard_block(in_channels, v)]
        in_channels = v

    self.conv = nn.Sequential(*layers)

    self.fc = nn.Sequential(nn.Dropout(),
                            nn.Linear(in_channels, fc_config[0]),
                            nn.ReLU(True),
                            nn.Dropout(),
                            nn.Linear(fc_config[0], fc_config[1]),
                            nn.ReLU(True),
                            nn.Linear(fc_config[1], num_classes), )

    # Initialize weights
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        # m.bias.data.zero_()

  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


def vgg16(num_classes=10):
  """VGG 16-layer model (configuration "D")"""
  return VGG([64, 64, 'M',
              128, 128, 'M',
              256, 256, 256, 'M',
              512, 512, 512, 'M',
              512, 512, 512, 'M'],
             [512, 512], num_classes)


if __name__ == '__main__':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)

  net = vgg16().cuda()
  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 32, 32).cuda())
  print(y.size())
