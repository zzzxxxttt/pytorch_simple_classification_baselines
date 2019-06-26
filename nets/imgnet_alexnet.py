import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
  def __init__(self, num_classes=1000):
    super(AlexNet, self).__init__()

    self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False)
    self.bn1 = nn.BatchNorm2d(96)

    self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, bias=False)
    self.bn2 = nn.BatchNorm2d(256)

    self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

    self.fc6 = nn.Linear(256 * 6 * 6, 4096)
    self.fc7 = nn.Linear(4096, 4096)
    self.logit = nn.Linear(4096, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    x = self.conv2(x)
    x = F.relu(self.bn2(x))
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    x = x.view(x.size(0), -1)
    x = F.dropout(x)
    x = F.relu(self.fc6(x))
    x = F.dropout(x)
    x = F.relu(self.fc7(x))
    x = self.logit(x)

    return x


def alexnet():
  return AlexNet()


if __name__ == '__main__':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)

  net = alexnet()

  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 224, 224))
  print(y.size())
