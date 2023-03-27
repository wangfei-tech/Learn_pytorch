import torchvision
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
train_set = torchvision.datasets.CIFAR10(root="./dataset",transform=torchvision.transforms.ToTensor(), train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",transform=torchvision.transforms.ToTensor(),train=False,download=True)


dataloader = DataLoader(train_set,batch_size=32)


class LearnConv2d(nn.Module):
    def __init__(self):
        super(LearnConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x
learnconv2d = LearnConv2d()

step =0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targets = data
    output  = learnconv2d(imgs)
    #print(imgs.shape)
    #print(output.shape)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("orign_image",imgs,step)
    writer.add_images("conv2d_show",output,step)
    step += 1
writer.close()