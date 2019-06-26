# Pytorch simple classification baselines

This repository contains simple pytorch version of LeNet-5(MNIST), ResNet(CIFAR, ImageNet), AlexNet(ImageNet), VGG-16(CIFAR, ImageNet) baselines.
There are both **nn.DataParallel** version and **nn.parallel.DistributedDataParallel** version for multi GPU training, I highly recommand using **nn.parallel.DistributedDataParallel** version since it's ~30% faster than using **nn.DataParallel**.     
 
## Requirements:
- python>=3.5
- pytorch>=1.1.0
- tensorboardX

## Train 

### single GPU or multi GPU using nn.DataParallel
* ```python mnist_train_eval.py ```
* ```python cifar_train_eval.py ```
* ```python imgnet_train_eval.py ```

### multi GPU using nn.parallel.DistributedDataParallel
* ```python  -m torch.distributed.launch --nproc_per_node 2 cifar_train_eval_dist.py --gpus 0,1```
* ```python  -m torch.distributed.launch --nproc_per_node 2 imgnet_train_eval_dist.py --gpus 0,1```


## Results:

### MNIST:
Model|Accuracy
:---:|:---:|
LeNet-5|99.26%

### CIFAR-10:
Model|Accuracy
:---:|:---:
ResNet-20|92.09%
ResNet-56|93.68%
VGG-16|93.99%

### ImageNet2012:
(todo)

