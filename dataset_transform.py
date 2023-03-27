import torchvision
from torch.utils.tensorboard import SummaryWriter


dataset_transfrom = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./dataset",transform=dataset_transfrom, train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",transform=dataset_transfrom,train=False,download=True)


# print(test_set[0])
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
writer = SummaryWriter("logs")
for i in range(10):
    img ,target = test_set[i]
    writer.add_image("dataset",img,i)
#print(test_set[0])
writer.close()