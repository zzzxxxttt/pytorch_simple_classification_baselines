import os
import time
import argparse
from tqdm import tqdm
from datetime import datetime
from PIL import ImageFile

import torch
import torch.optim as optim
import torchvision.datasets as datasets

from tensorboardX import SummaryWriter

from nets.imgnet_alexnet import *
from nets.imgnet_resnet import *
from nets.imgnet_vgg import *

from utils.preprocessing import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='classification_baselines')

parser.add_argument('--local_rank', dest='local_rank', type=int, default=0)

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='alexnet_baseline')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=5e-4)

parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--use_gpu', type=str, default='3')
parser.add_argument('--num_workers', type=int, default=20)

cfg = parser.parse_args()

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu


def main():
  torch.cuda.set_device(cfg.local_rank)
  torch.distributed.init_process_group(backend='nccl', init_method='env://')

  print('Prepare dataset ...')
  traindir = os.path.join(cfg.data_dir, 'train')
  train_dataset = datasets.ImageFolder(traindir, imagenet_transform(is_training=True))
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.train_batch_size,
                                             shuffle=True,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True,
                                             sampler=train_sampler)

  evaldir = os.path.join(cfg.data_dir, 'val')
  val_dataset = datasets.ImageFolder(evaldir, imagenet_transform(is_training=False))
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=cfg.test_batch_size,
                                           shuffle=False,
                                           num_workers=cfg.num_workers,
                                           pin_memory=True)

  # create model
  print("=> creating model alexnet")
  model = resnet18().cuda()
  model = nn.parallel.DistributedDataParallel(model,
                                              device_ids=[cfg.local_rank, ],
                                              output_device=cfg.local_rank)

  optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedulr = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90], 0.1)
  criterion = torch.nn.CrossEntropyLoss()

  summary_writer = SummaryWriter(cfg.log_dir)

  def train(epoch):
    # switch to train mode
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):

      # compute output
      outputs = model(inputs.cuda())
      loss = criterion(outputs, targets.cuda())

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if cfg.local_rank == 0 and batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def validate(epoch):
    # switch to evaluate mode
    model.eval()
    top1 = 0
    top5 = 0
    with torch.no_grad():
      for i, (inputs, targets) in tqdm(enumerate(val_loader)):
        # compute output
        output = model(inputs.cuda())

        # measure accuracy and record loss
        _, pred = output.data.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.cuda().view(1, -1).expand_as(pred))

        top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
        top5 += correct[:5].view(-1).float().sum(0, keepdim=True).item()

    top1 *= 100 / len(val_dataset)
    top5 *= 100 / len(val_dataset)
    print('%s Precision@1 ==> %.2f%%  Precision@1: %.2f%%\n' % (datetime.now(), top1, top5))

    summary_writer.add_scalar('Precision@1', top1, epoch)
    summary_writer.add_scalar('Precision@5', top5, epoch)
    return

  for epoch in range(cfg.max_epochs):
    lr_schedulr.step(epoch)
    train(epoch)
    if cfg.local_rank == 0:
      validate(epoch)
      torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))
      print('checkpoint saved to %s !' % os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))

  summary_writer.close()


if __name__ == '__main__':
  main()
