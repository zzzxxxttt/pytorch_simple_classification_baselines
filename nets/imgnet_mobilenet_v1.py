import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
  def __init__(self, in_planes, out_planes, stride=1):
    super(Block, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(out_planes)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    return out


class MobileNet(nn.Module):
  def __init__(self, conv_cfg, num_classes=10):
    super(MobileNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)

    layers = []
    in_planes = 32
    for x in conv_cfg:
      out_planes = x if isinstance(x, int) else x[0]
      stride = 1 if isinstance(x, int) else x[1]
      layers.append(Block(in_planes, out_planes, stride))
      in_planes = out_planes

    self.conv = nn.Sequential(*layers)
    self.fc = nn.Linear(1024, num_classes)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.conv(out)
    out = out.mean(2).mean(2)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out


def mobilenet_v1():
  # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
  return MobileNet(conv_cfg=[64,
                             (128, 2), 128,
                             (256, 2), 256,
                             (512, 2), 512, 512, 512, 512, 512,
                             (1024, 2), 1024])


if __name__ == '__main__':
  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = mobilenet_v1()
  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 224, 224))
  print(y.size())
