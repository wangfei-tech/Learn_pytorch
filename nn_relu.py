import torch
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch import nn
import torchvision
from  torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
dataset = torchvision.datasets.CIFAR10(root="./dataset",transform=torchvision.transforms.ToTensor(), train=True,download=True)
dataloader = DataLoader(dataset, batch_size=64,num_workers=2)
input = torch.Tensor([[1, -1],
                       [2, -1]])
input = torch.reshape(input,(-1, 1, 2, 2))
class LearnRelu(nn.Module):
    def __init__(self):
        super(LearnRelu, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self,x):
        #output = self.relu(x)
        output = self.sigmoid(x)
        return output

learnrelu = LearnRelu()

#output  = learnrelu(input)
#print(output)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, target = data
    imgs = learnrelu(imgs)
    writer.add_images("nn_sigmoid",imgs,step)
    step += 1


writer.close()



        
