import os

root_dir = "/home/wf/learn_torch/data/train"
target_dir_ants = "ants_image"
target_dir_bees = "bees_image"

img_path = os.listdir(os.path.join(root_dir,target_dir_ants))
ants_label = target_dir_ants.split("_")[0]
out_dir = "ants_label"
for i in img_path:
    file_name = i.split(".jpg")[0]
    with open(os.path.join(root_dir,out_dir,"{}.txt".format(file_name)),"w") as f:
        f.write(ants_label)
img_path = os.listdir(os.path.join(root_dir,target_dir_bees))
bees_label = target_dir_bees.split("_")[0]
out_dir = "bees_label"
for i in img_path:
    file_name = i.split(".jpg")[0]
    with open(os.path.join(root_dir,out_dir,"{}.txt".format(file_name)),"w") as f:
        f.write(bees_label)