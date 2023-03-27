import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from model import NN_Model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

img_path_dir = "/home/wf/learn_torch/dataset/ship.png"

label = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")

image = Image.open(img_path_dir)

print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),torchvision.transforms.ToTensor()])

image = transform(image)

nn_model = NN_Model()

model = torch.load("/home/wf/learn_torch/models/nn_model: 65.pth")
print(model)
image = torch.reshape(image,(1,3,32,32))
image = image.to(device)
model.eval()
with torch.no_grad():
    output = model(image)
    output = output.argmax(1)
print(output)

#data_tran  = torchvision.transforms.Compose([int])

#data = data_tran(output)
#print(data)

print(label[output])