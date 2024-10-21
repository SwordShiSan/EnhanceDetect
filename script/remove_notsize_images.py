import cv2
import os
import shutil
from tqdm import tqdm

# vis_root = "/mnt/nas/lailihao/M3FD/Vis/"
# images = os.listdir(vis_root)

# for image in images:
#     img = cv2.imread(vis_root + image)
#     if img.shape != (768, 1024, 3):
#         print(image, ":", img.shape)

vis_root = "/mnt/nas/lailihao/M3FD/Vis/"
inf_root = "/mnt/nas/lailihao/M3FD/Inf/"
labels_root = "/mnt/nas/lailihao/M3FD/labels/"

save_vis_root = "/mnt/nas/lailihao/M3FD_final/vis/"
save_inf_root = "/mnt/nas/lailihao/M3FD_final/inf/"
save_labels_root = "/mnt/nas/lailihao/M3FD_final/labels/"

images = os.listdir(vis_root)
remove_images = []

for image in tqdm(images):
    img = cv2.imread(vis_root + image)
    if img.shape != (768, 1024, 3):
        print(image, ":", img.shape)
        remove_images.append(image)

save_images = list(set(images) - set(remove_images))
#
# for image in tqdm(save_images):
#     shutil.copy(os.path.join(vis_root, image), os.path.join(save_vis_root, image))
#     shutil.copy(os.path.join(inf_root, image), os.path.join(save_inf_root, image))
#     shutil.copy(os.path.join(labels_root, image[:-4]+".txt"), os.path.join(save_labels_root, image[:-4]+".txt"))

# 从train.txt,val.txt删除
with open("/mnt/nas/lailihao/M3FD/train.txt", "r") as f:
    ori_train_list = f.readlines()
    ori_train_list = [line.rstrip() for line in ori_train_list]
    cur_train_list = list(set(ori_train_list) - set(remove_images))
    with open("/mnt/nas/lailihao/M3FD_final/train.txt", "w") as f2:
        for image in tqdm(cur_train_list):
            f2.write(image+"\n")

with open("/mnt/nas/lailihao/M3FD/val.txt", "r") as f:
    ori_val_list = f.readlines()
    ori_val_list = [line.rstrip() for line in ori_val_list]
    cur_val_list = list(set(ori_val_list) - set(remove_images))
    with open("/mnt/nas/lailihao/M3FD_final/val.txt", "w") as f2:
        for image in tqdm(cur_val_list):
            f2.write(image+"\n")

'''
01495.png : (480, 800, 3)
01496.png : (480, 800, 3)
01497.png : (480, 800, 3)
01498.png : (480, 800, 3)
01499.png : (480, 800, 3)
01501.png : (480, 800, 3)
01503.png : (480, 800, 3)
01504.png : (480, 800, 3)
01505.png : (480, 800, 3)
01506.png : (480, 800, 3)
01508.png : (480, 800, 3)
01510.png : (480, 800, 3)
01511.png : (480, 800, 3)
01512.png : (480, 800, 3)
01515.png : (480, 800, 3)
01516.png : (480, 800, 3)
01517.png : (480, 800, 3)
01520.png : (480, 800, 3)
01521.png : (480, 800, 3)
01523.png : (480, 800, 3)
01524.png : (480, 800, 3)
01525.png : (480, 800, 3)
01526.png : (480, 800, 3)
01527.png : (480, 800, 3)
01528.png : (480, 800, 3)
02890.png : (520, 880, 3)
02891.png : (520, 880, 3)
02892.png : (520, 880, 3)
02893.png : (520, 880, 3)
02895.png : (520, 880, 3)
02898.png : (520, 880, 3)
02899.png : (520, 880, 3)
02900.png : (520, 880, 3)
02902.png : (520, 880, 3)
02905.png : (520, 880, 3)
02906.png : (520, 880, 3)
02907.png : (520, 880, 3)
02908.png : (520, 880, 3)
02909.png : (520, 880, 3)
02912.png : (520, 880, 3)
02913.png : (520, 880, 3)
02917.png : (520, 880, 3)
02918.png : (520, 880, 3)
02919.png : (520, 880, 3)
02920.png : (520, 880, 3)
02923.png : (520, 880, 3)
02924.png : (520, 880, 3)
02925.png : (520, 880, 3)
02926.png : (520, 880, 3)
02928.png : (520, 880, 3)
02929.png : (520, 880, 3)
02930.png : (520, 880, 3)
02932.png : (520, 880, 3)
01494.png : (480, 800, 3)
01500.png : (480, 800, 3)
01502.png : (480, 800, 3)
01507.png : (480, 800, 3)
01509.png : (480, 800, 3)
01513.png : (480, 800, 3)
01518.png : (480, 800, 3)
01522.png : (480, 800, 3)
02896.png : (520, 880, 3)
02901.png : (520, 880, 3)
02911.png : (520, 880, 3)
02915.png : (520, 880, 3)
02922.png : (520, 880, 3)
02927.png : (520, 880, 3)
02933.png : (520, 880, 3)
02938.png : (520, 880, 3)
01514.png : (480, 800, 3)
01519.png : (480, 800, 3)
01529.png : (480, 800, 3)
02894.png : (520, 880, 3)
02897.png : (520, 880, 3)
02910.png : (520, 880, 3)
02914.png : (520, 880, 3)
02916.png : (520, 880, 3)
02921.png : (520, 880, 3)
02931.png : (520, 880, 3)
'''
