import torch

input = torch.tensor([[0.1, 0.2],
                     [0.3, 0.4]])


print(input.argmax(0))#竖着看
print(input.argmax(1))#横着看

pred = input.argmax(1)

targets = torch.tensor([0, 1])

print((pred == targets).sum())