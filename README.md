# Pytorch simple classification baselines

This repository contains simple pytorch version of LeNet-5(MNIST), ResNet(CIFAR, ImageNet), AlexNet(ImageNet), VGG-16(CIFAR, ImageNet) baselines.
There are both **nn.DataParallel** and **nn.parallel.DistributedDataParallel** version for multi GPU training, I highly recommand using nn.parallel.DistributedDataParallel since it's considerably faster than using nn.DataParallel.     
 
## Requirements:
- python>=3.5
- pytorch>=0.4.1(>=1.1.0 for DistributedDataParallel version)
- tensorboardX(optional)

## Train 

### single GPU or multi GPU using nn.DataParallel
* ```python mnist_train_eval.py ```
* ```python cifar_train_eval.py ```
* ```python imgnet_train_eval.py ```

### multi GPU using nn.parallel.DistributedDataParallel
* ```python  -m torch.distributed.launch --nproc_per_node 2 cifar_train_eval.py --dist --gpus 0,1```
* ```python  -m torch.distributed.launch --nproc_per_node 2 imgnet_train_eval.py --dist --gpus 0,1```


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
Model|Top-1 Accuracy|Top-5 Accuracy
:---:|:---:|:---:
ResNet-18|69.67%|89.29%

