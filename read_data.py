import os
import cv2 
from torch.utils.data import Dataset
from PIL import Image
class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        # print(self.path)
        self.img_path = os.listdir(self.path)
        self.img_path = sorted(self.img_path)
        # print(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        print(img_name)
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = cv2.imread(img_item_path, cv2.IMREAD_UNCHANGED)
        cv2.imshow("show_img", img)
        cv2.waitKey(0)
        img1 = Image.open(img_item_path)
        img1.show()
        label = self.label_dir
        return img ,label

    def __len__(self):
        return len(self.img_path)

root_dir = "/home/wf/learn_torch/data/train"
ant_label_dir = "ants"
ants_dataset = MyData(root_dir, ant_label_dir)
bees_label_dir = "bees"
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset

img, label = bees_dataset[0]
# print(img,label)
