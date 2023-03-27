import torch.nn as nn
import torch.nn.functional as F
import torch

class LearnNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output  = input +1
        return output

learnNN = LearnNN()

x = torch.tensor(1.0)

output = learnNN(x)

print(output)