import torch.nn as nn
import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
#定义一个神经网络模型
class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model,self).__init__()
        #self.device = device
        self.sequential = nn.Sequential(nn.Conv2d(3,32,5, padding=2),
                          nn.MaxPool2d(2),
                          nn.Conv2d(32, 32, 5, stride=1, padding=2),
                          nn.MaxPool2d(2),
                          nn.Conv2d(32, 64, kernel_size=5, padding=2),
                          nn.MaxPool2d(2),
                          nn.Flatten(),
                          nn.Linear(1024,64),
                          nn.Linear(64, 10))
    def forward(self, input):
        #input.to(self.device)
        output = self.sequential(input)
        return output