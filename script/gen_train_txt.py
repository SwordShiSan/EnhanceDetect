import os
import random

# vis_root = "/mnt/nas/lailihao/M3FD/Vis/"
# names = os.listdir(vis_root)
# print(len(names))
# train_names = random.sample(names, 3200)
# test_names = list(set(names) - set(train_names))
#
# with open("/mnt/nas/lailihao/M3FD/train.txt", "w") as f:
#     for name in train_names:
#         f.write(name[:-4] + '\n')
#
# with open("/mnt/nas/lailihao/M3FD/test.txt", "w") as f:
#     for name in test_names:
#         f.write(name[:-4] + '\n')

# add suffix
with open("/mnt/nas/lailihao/M3FD/train.txt", "w") as f:
    with open("/mnt/nas/lailihao/M3FD/train_ori.txt", "r") as oir_f:
        names = oir_f.readlines()

    for item in names:
        file_name = item.strip()
        f.write(file_name + ".png" + "\n")

with open("/mnt/nas/lailihao/M3FD/val.txt", "w") as f:
    with open("/mnt/nas/lailihao/M3FD/val_ori.txt", "r") as oir_f:
        names = oir_f.readlines()

    for item in names:
        file_name = item.strip()
        f.write(file_name + ".png" + "\n")
