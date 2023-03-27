import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn
from torch.nn import Linear
from torch.nn import Module

dataset = torchvision.datasets.CIFAR10(root="./dataset",transform=torchvision.transforms.ToTensor(), train=True,download=True)
dataloader = DataLoader(dataset, batch_size=64,num_workers=2)

class LearnLinear(Module):
    def __init__(self):
        super(LearnLinear,self).__init__()
        self.linear = Linear(196608,60)
    def forward(self,x):
        output = self.linear(x)
        return output

learnlinear = LearnLinear()

for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = learnlinear(output)
    print(output.shape)