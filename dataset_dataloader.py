import torchvision
#准备测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
test_dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=True,num_workers=3,drop_last=False)

#测试数据集中的地一张图片以及标签
img,target = test_dataset[0]
print(img.shape)
print(target)

writer =SummaryWriter("logs")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs)
        # print(target)
        writer.add_images("Epoch: {}".format(epoch),imgs,step)
        step += 1

writer.close()