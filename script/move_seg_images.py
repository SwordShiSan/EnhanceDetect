import os
import shutil
from tqdm import tqdm

ori_msrs_train_root_vi = "/mnt/nas/lailihao/MSRS_ori/train/vi"
ori_msrs_train_root_ir = "/mnt/nas/lailihao/MSRS_ori/train/ir"
ori_msrs_train_root_labels = "/mnt/nas/lailihao/MSRS_ori/train/Segmentation_labels/"

ori_msrs_test_root_vi = "/mnt/nas/lailihao/MSRS_ori/test/vi"
ori_msrs_test_root_ir = "/mnt/nas/lailihao/MSRS_ori/test/ir"
ori_msrs_test_root_labels = "/mnt/nas/lailihao/MSRS_ori/test/Segmentation_labels/"

save_vis_root = "/mnt/nas/lailihao/MSRS/vis/"
save_inf_root = "/mnt/nas/lailihao/MSRS/inf/"
save_label_root = "/mnt/nas/lailihao/MSRS/labels/"

train_images = os.listdir(ori_msrs_train_root_vi)
test_images = os.listdir(ori_msrs_test_root_vi)

for name in tqdm(train_images):
    shutil.copy(os.path.join(ori_msrs_train_root_vi, name), os.path.join(save_vis_root, name))
    shutil.copy(os.path.join(ori_msrs_train_root_ir, name), os.path.join(save_inf_root, name))
    shutil.copy(os.path.join(ori_msrs_train_root_labels, name), os.path.join(save_label_root, name))

for name in tqdm(test_images):
    shutil.copy(os.path.join(ori_msrs_test_root_vi, name), os.path.join(save_vis_root, name))
    shutil.copy(os.path.join(ori_msrs_test_root_ir, name), os.path.join(save_inf_root, name))
    shutil.copy(os.path.join(ori_msrs_test_root_labels, name), os.path.join(save_label_root, name))

with open("/mnt/nas/lailihao/MSRS/train.txt", "w") as f:
    for name in train_images:
        f.write(name + "\n")

with open("/mnt/nas/lailihao/MSRS/test.txt", "w") as f:
    for name in test_images:
        f.write(name + "\n")
