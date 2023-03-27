""""
１．计算实际输出和目标之间的差距
２．为更新输出提供依据(反向传播　　)
"""
import torch
import torchvision
import  torch.nn as nn
from torch.nn import L1Loss, MSELoss


input = torch.tensor([1, 2, 3],dtype=torch.float)
target = torch.tensor([1, 3, 5], dtype=torch.float)

# loss = L1Loss()
# result = loss(input,target)

# loss_mse = MSELoss()
# result = loss_mse(input, target)
# print(result)

x = torch.tensor([0.1, 0.2, 0.3])
x = torch.reshape(x, (1, 3))

y = torch.tensor([1])

loss_cross = nn.CrossEntropyLoss()
result = loss_cross(x,y)
print(result)