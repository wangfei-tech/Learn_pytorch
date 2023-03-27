import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)


#保存方式1
#torch.save(vgg16, "/home/wf/learn_torch/models/vgg16.pth")
#保存方式２
torch.save(vgg16.state_dict(), "/home/wf/learn_torch/models/vgg16_dict.pth")

# torch.load()
#model = torch.load("/home/wf/learn_torch/models/vgg16.pth")
#print(model)
model = torch.load("/home/wf/learn_torch/models/vgg16_dict.pth")
print(model)