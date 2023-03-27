import torch
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
import numpy as np
import torch
from PIL import Image
writer = SummaryWriter("logs")
img_path = "/home/wf/learn_torch/data/train/bees_image/92663402_37f379e57a.jpg"

img = cv.imread(img_path) #BGR 的显示格式
# cv.imshow("show_img",img)
# cv.waitKey(0)
b,g,r = cv.split(img) 
img = cv.merge([r,g,b])
print(type(img))
print(img.shape)

# img = Image.open(img_path)
# img_arr = np.array(img)
# print(type(img_arr))
# print(img_arr.shape)
writer.add_image("test",img,1,dataformats="HWC") #RGB的显示格式　所以要转换

for i in range(99):
    writer.add_scalar("y=x", 3*i, i)
# writer.add_scalar()

writer.close()
