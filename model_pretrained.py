import torchvision
import torch.nn as nn
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True) #下载训练好的模型
#print(vgg16_true)
#print("ok!")
train_set = torchvision.datasets.CIFAR10(root="./dataset",transform=torchvision.transforms.ToTensor(), train=True,download=True)

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6].add_module("add_linear1", nn.Linear(1000,10))
print(vgg16_false)