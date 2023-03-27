from torchvision import transforms
from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
"""
类似与一个工具箱　将数据类型装换成ｔｅｎｓｏｒ类型
图片-->工具--->结果
tensor 数据类型 包装了神经网络需要的类型
`
输入　* PIL     * Image.open()
输出　* tensor  * ToTensor
作用　* narrays * cv.imread()
`
"""
"""
利用transforms.ToTensor去看两个问题
"""


img_path = "/home/wf/learn_torch/data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

#ToTensor
tensor_tran = transforms.ToTensor()
tensor_img = tensor_tran(img)
#print(tensor_img)

# Normalize
print(tensor_img[0][0][0])
trans_normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normal  = trans_normal(tensor_img)
print(img_normal[0][0][0])

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
print(type(img_resize))
img_resize = tensor_tran(img)
print(type(img_resize))

#Compose -resize 
trains_resize_2 = transforms.Resize(512)
trains_compose = transforms.Compose([trains_resize_2,tensor_tran])
trains_resize_2 = trains_compose(img)

#RandomCrop
tran_random = transforms.RandomResizedCrop(250)
tran_compose = transforms.Compose([tran_random,tensor_tran])
tran_random  = tran_compose(img)


writer = SummaryWriter("logs")
writer.add_image("Tensor",tensor_img)
writer.add_image("Tensor",img_normal,2)
writer.add_image("Tensor",img_resize,3)
writer.add_image("Tensor",trains_resize_2,4)
writer.add_image("Tensor",tran_random,5)


writer.close()

