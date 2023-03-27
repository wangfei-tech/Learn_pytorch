"""
神经网络模型

数据（输入\标注）

损失函数

cuda
"""
import time
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from model import *
from torch.utils.tensorboard import SummaryWriter

#定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_data = torchvision.datasets.CIFAR10(root="/home/wf/learn_torch/dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data  = torchvision.datasets.CIFAR10(root="/home/wf/learn_torch/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_dataloader = DataLoader(train_data,batch_size=64,num_workers=0)
test_dataloader = DataLoader(test_data,batch_size=64, num_workers=0)

train_data_size = len(train_data)
test_data_size  = len(test_data)
print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

#神经网络
nn_model = NN_Model()
nn_model.to(device)
#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
#优化器
learning_rate= 0.01

optimizer = torch.optim.SGD(nn_model.parameters(),lr=learning_rate)

#设置训练网络的参数
#记录训练的次数
total_train_step = 0
#记录测试次数
total_test_step = 0
#训练轮数
epoch = 45
#添加tensorboard
writer = SummaryWriter("logs")
#start_time = time.time()
for i in range(epoch):
    #print(30*('*'))
    print("------第{}轮训练开始------".format(i+1))
    start_time = time.time()
    nn_model.train()
    for data in train_dataloader:
        imgs, target = data
        imgs = imgs.to(device)
        target = target.to(device)
        output = nn_model(imgs)
        #print("label: ",target)
        loss = loss_fn(output,target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_train_step +=1
        #writer.add_scalar("train_loss",loss.item(),total_train_step)
        if total_train_step%100 == 0:
            end_time = time.time()
            print((end_time-start_time))
            print("训练次数: {}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    total_test_loss = 0.0
    totai_accuracy = 0.0
    nn_model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, target = data
            imgs = imgs.to(device)
            target = target.to(device)
            output = nn_model(imgs)
            loss = loss_fn(output,target)
            #total_test_step  = total_test_step +1
            total_test_loss = total_test_loss + loss.item()
            accuracy = (output.argmax(1) == target).sum()
            totai_accuracy = accuracy +totai_accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(totai_accuracy/test_data_size))
    writer.add_scalar("test_loss",loss.item(),total_test_step)
    total_test_step  = total_test_step +1
    torch.save(nn_model, "/home/wf/learn_torch/models/nn_model: {}.pth".format(int (100*(totai_accuracy/test_data_size))))

if __name__ == "__mian__":
    #实例化神经网络
    nn_model =NN_Model()
    input = torch.ones((64, 3, 32, 32))
    output = nn_model(input)
    print(output.shape)
