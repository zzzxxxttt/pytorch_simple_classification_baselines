import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 20, kernel_size=5, bias=False)
    self.conv2 = nn.Conv2d(20, 50, kernel_size=5, bias=False)
    self.fc1 = nn.Linear(800, 500)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    out = self.conv1(x)
    out = F.relu(F.max_pool2d(out, 2))
    out = self.conv2(out)
    out = F.relu(F.max_pool2d(out, 2))
    out = out.view(-1, 800)
    out = self.fc1(out)
    out = F.relu(out)
    out = self.fc2(out)
    return out


if __name__ == '__main__':
  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = LeNet()
  for m in net.modules():
    m.register_forward_hook(hook)

  y = net(torch.randn(1, 1, 28, 28))
  print(y.size())
