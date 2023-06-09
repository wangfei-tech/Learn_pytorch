import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear,Sequential
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
class Cfar10_model(nn.Module):
    def __init__(self):
        super(Cfar10_model,self).__init__()
        # self.conv1 = Conv2d(3,32,5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, stride=1, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, kernel_size=5, padding=2)
        # self.maxpool3  =MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024,64)
        # self.linear2 = Linear(64, 10)
        self.sequential = Sequential(Conv2d(3,32,5, padding=2),
                          MaxPool2d(2),
                          Conv2d(32, 32, 5, stride=1, padding=2),
                          MaxPool2d(2),
                          Conv2d(32, 64, kernel_size=5, padding=2),
                          MaxPool2d(2),
                          Flatten(),
                          Linear(1024,64),
                          Linear(64, 10))
    
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.sequential(x)
        return x

cfar10_model = Cfar10_model()
dataset = torchvision.datasets.CIFAR10(root="./dataset",transform=torchvision.transforms.ToTensor(), train=True,download=True)
dataloader = DataLoader(dataset, batch_size=64,num_workers=2)

writer = SummaryWriter("logs")

for data in dataloader:
    imgs, target = data
    output = cfar10_model(imgs)
    print(output.shape)
    writer.add_graph(cfar10_model,imgs)

writer.close()



