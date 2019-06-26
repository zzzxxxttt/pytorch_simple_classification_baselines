import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InvertedResidual(nn.Module):
  def __init__(self, inplanes, planes, stride, expand_ratio):
    super(InvertedResidual, self).__init__()
    self.skip = stride == 1 and inplanes == planes
    hidden_dim = round(inplanes * expand_ratio)

    # pw
    self.conv1 = \
      nn.Sequential(nn.Conv2d(inplanes, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)) \
        if expand_ratio > 1 else nn.Sequential()
    # dw
    self.conv2 = \
      nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True))
    # pw-linear
    self.conv3 = \
      nn.Sequential(nn.Conv2d(hidden_dim, planes, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(planes))

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)

    if self.skip:
      out = x + out

    return out


class MobileNetV2(nn.Module):
  def __init__(self, block=InvertedResidual, n_class=1000, input_size=224, width_mult=1.):
    super(MobileNetV2, self).__init__()
    input_channel = int(32 * width_mult)
    last_channel = int(1280 * width_mult)
    # [expand_ratio, channel, num_blocks, stride]
    interverted_residual_setting = [[1, 16, 1, 1],
                                    [6, 24, 2, 2],
                                    [6, 32, 3, 2],
                                    [6, 64, 4, 2],
                                    [6, 96, 3, 1],
                                    [6, 160, 3, 2],
                                    [6, 320, 1, 1]]

    # building first layer
    self.conv_first = nn.Conv2d(3, input_channel, 3, 2, 1, bias=False)
    self.bn_first = nn.BatchNorm2d(input_channel)

    # building inverted residual blocks
    self.features = nn.ModuleList()
    for t, c, n, s in interverted_residual_setting:
      output_channel = int(c * width_mult)
      for i in range(n):
        self.features.append(block(input_channel, output_channel, s if i == 0 else 1, expand_ratio=t))
        input_channel = output_channel

    # building last several layers
    self.conv_last = nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False)
    self.bn_last = nn.BatchNorm2d(last_channel)

    # building classifier
    self.classifier = nn.Sequential(nn.Dropout(0.2),
                                    nn.Linear(last_channel, n_class))

    self._initialize_weights()

  def forward(self, x):
    x = self.conv_first(x)
    x = F.relu6(self.bn_first(x), inplace=True)

    for block in self.features:
      x = block(x)

    x = self.conv_last(x)
    x = F.relu6(self.bn_last(x), inplace=True)

    x = x.mean(3).mean(2)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


if __name__ == '__main__':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)


  net = MobileNetV2()
  for m in net.modules():
    m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 224, 224))
  print(y.size())
