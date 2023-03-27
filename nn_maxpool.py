import torch
import torch.nn
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
"""
最大池化的作用:降低数据量
"""


input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]],dtype=float)

input = torch.reshape(input, (-1, 1, 5, 5))

dataset = torchvision.datasets.CIFAR10(root="./dataset",transform=torchvision.transforms.ToTensor(), train=True,download=True)
dataloader = DataLoader(dataset, batch_size=64)
# print(input.shape)

writer = SummaryWriter("logs")

class LearnMaxpool(nn.Module):
    def __init__(self):
        super(LearnMaxpool,self).__init__()
        self.maxplool = MaxPool2d(kernel_size=2, ceil_mode=False)
    def forward(self, x):
        output = self.maxplool(x)
        return output

learnmaxpool = LearnMaxpool()
#output = learnmaxpool(input)
#print(output)
step = 0
for data in dataloader:

    imgs, target = data
    writer.add_images("pool_shows", imgs,step)
    imgs1 = learnmaxpool(imgs)
    writer.add_images("after_pool",imgs1, step)
    step += 1
writer.close()

