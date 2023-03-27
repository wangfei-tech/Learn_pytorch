import torch
import torch.nn as nn
from torch.nn import Flatten, MaxPool2d, Sequential,Linear, Conv2d
from torch.utils.data import DataLoader
import torchvision

dataset = torchvision.datasets.CIFAR10(root="./dataset",transform=torchvision.transforms.ToTensor(), train=True,download=True)
dataloader = DataLoader(dataset, batch_size=1,num_workers=2)

class Cfar10_model(nn.Module):
    def __init__(self):
        super(Cfar10_model,self).__init__()
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
        x = self.sequential(x)
        return x

cfar10_model = Cfar10_model()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(cfar10_model.parameters(),lr=0.1)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, target = data
        outputs = cfar10_model(imgs)
        results_loss = loss(outputs,target)
        # results.backward()
        optim.zero_grad()
        results_loss.backward()
        optim.step()
        running_loss = running_loss + results_loss
        # print("outputs:",outputs)
        # print("targets:",target)
        # print("results:",results_loss)
        # print("********************\n")
        # print(outputs.shape)
    print(running_loss)