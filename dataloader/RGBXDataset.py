import os
from pickletools import uint8
import cv2
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._det_label_path = setting['det_label_root']
        self._det_label_format = setting['det_label_format']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]
        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        x_path = os.path.join(self._x_path, item_name + self._x_format)
        gt_path = os.path.join(self._gt_path, item_name + self._gt_format)
        det_label_path = os.path.join(self._det_label_path, item_name + self._det_label_format)

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, "RGB")

        if self._x_single_channel:
            x = self._open_image(x_path, "Gray")
            x = cv2.merge([x, x, x])
        else:
            x = self._open_image(x_path, "RGB")

        # seg_label = self._open_image(gt_path, "Gray", dtype=np.uint8)
        # if self._transform_gt:
        #     seg_label = self._gt_transform(seg_label)
        seg_label = np.zeros((rgb.shape[0], rgb.shape[1]))

        # Detr targets 是coco格式,"bbox": [x, y, width, height],"category_id": int,
        # 需要target["boxes"],target["labels"]
        target = self._read_detect_label(det_label_path)
        target["orig_size"] = torch.tensor([rgb.shape[0], rgb.shape[1]])
        # target["image_id"] = torch.tensor([index])
        target["image_id"] = int(item_name)

        if self.preprocess is not None and self._split_name == 'train':
            rgb, seg_label, x, Mask = self.preprocess(rgb, seg_label, x)
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            seg_label = torch.from_numpy(np.ascontiguousarray(seg_label)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()
            Mask = torch.from_numpy(np.ascontiguousarray(Mask)).float()

        if self.preprocess is not None and self._split_name == 'val':
            rgb, seg_label, x = self.preprocess(rgb, seg_label, x)

        # target["boxes"],target["labels"]
        # 这样出去batch只能为1,因为每个shape不一致,而且dataloader会自动给target的值覆上一层tensor
        # 希望输出的targets= batch traget
        if self.preprocess is not None and self._split_name == 'train':
            output_dict = dict(data=rgb, seg_label=seg_label, det_label=target, modal_x=x, Mask=Mask, fn=str(item_name),
                               n=len(self._file_names))
        else:
            output_dict = dict(data=rgb, seg_label=seg_label, det_label=target, modal_x=x, fn=str(item_name),
                               n=len(self._file_names))
        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode="Gray", dtype=np.float32):
        if mode == "RGB":
            img = np.array(cv2.imread(filepath, cv2.IMREAD_COLOR), dtype=dtype)
            img = img[:, :, [2, 1, 0]]  # BGR to RGB
        elif mode == "Gray":
            img = np.array(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), dtype=dtype)
        else:
            img = np.array(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), dtype=dtype)

        return img

    @staticmethod
    def _gt_transform(seg_label):
        return seg_label - 1

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    def _read_detect_label(self, det_label_path):
        target = np.loadtxt(str(det_label_path), dtype=np.float32)
        # [n*[cls, cx, cy, w, h]] n是gt框个数
        labels = torch.from_numpy(target).view(-1, 5)

        # detr 网络输入?;yolo网络输入cx cy w h
        boxes = labels[:, 1:].to(dtype=torch.float32)  # n cx cy w h
        # classes = labels[:, 0:1].to(dtype=torch.int64)  # [n cls] 得是[n*cls]
        classes = (labels[:, 0:1].to(dtype=torch.int64)).view(-1)  # [n*cls]

        # TODO
        # boxes[:, 2:] += boxes[:, :2]
        # boxes[:, 0::2].clamp_(min=0, max=w)
        # boxes[:, 1::2].clamp_(min=0, max=h)

        # keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        # boxes = boxes[keep]
        # classes = classes[keep]

        # print(labels)
        detect_label = {"boxes": boxes, "labels": classes}
        return detect_label


class DetectDataset(data.Dataset):
    """
    dataset: contain detect infrared and visible dataset,construct like this
        detect root:
            vis
            inf
            labels(detect,yolo format .txt)
            train.txt(with suffix)
            val.txt
    Args:
        setting(dict): detect_root--Absolute path to root;x_single_channel--infrared channel true for 1 else 3
        split_name(str): train/val
        is_preprocess(bool): train/val data preprocess
    """

    def __init__(self, setting, split_name="train", is_preprocess=True):
        super(DetectDataset, self).__init__()
        self.is_preprocess = is_preprocess
        self._split_name = split_name
        # TODO: Vis,Inf -> vis,inf
        self._rgb_path = "vis"
        self._x_path = "inf"
        self._label_path = "labels"
        self._detect_root = setting['detect_root']
        self._x_single_channel = setting['x_single_channel']
        self._detect_file_names = self._get_file_names(self._detect_root, self._split_name)
        self.image_resize = setting['image_resize']

    def __len__(self):
        return len(self._detect_file_names)

    def __getitem__(self, index):
        image_name = self._detect_file_names[index]  # file.png with suffix
        rgb_path = os.path.join(self._detect_root, self._rgb_path, image_name)
        x_path = os.path.join(self._detect_root, self._x_path, image_name)
        det_label_path = os.path.join(self._detect_root, self._label_path, image_name[:-4] + ".txt")

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, "RGB")

        if self._x_single_channel:
            x = self._open_image(x_path, "Gray")
            x = cv2.merge([x, x, x])
        else:
            x = self._open_image(x_path, "RGB")

        # Detr det_label 是coco格式,"bbox": [x, y, width, height],"category_id": int,
        # det_label["boxes"],det_label["labels"],det_label["orig_size"],det_label["image_id"]
        det_label = self._read_detect_label(det_label_path)
        det_label["orig_size"] = torch.tensor([rgb.shape[0], rgb.shape[1]])
        det_label["image_id"] = int(image_name[:-4])  # image name is image_id

        if self.is_preprocess:
            if self._split_name == 'train':
                rgb, det_label, x, Mask = self._preprocess_train(rgb, det_label, x)
            elif self._split_name == 'val':
                rgb, det_label, x = self._preprocess_val(rgb, det_label, x)

        if self.is_preprocess and self._split_name == 'train':
            output_dict = dict(data=rgb, det_label=det_label, modal_x=x, Mask=Mask, fn=str(image_name))
        else:
            output_dict = dict(data=rgb, det_label=det_label, modal_x=x, fn=str(image_name))
        return output_dict

    @staticmethod
    def _get_file_names(data_root, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(data_root, split_name + ".txt")

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode="Gray", dtype=np.float32):
        if mode == "RGB":
            img = np.array(cv2.imread(filepath, cv2.IMREAD_COLOR), dtype=dtype)
            img = img[:, :, [2, 1, 0]]  # BGR to RGB
        elif mode == "Gray":
            img = np.array(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), dtype=dtype)
        else:
            img = np.array(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), dtype=dtype)

        return img

    def _preprocess_train(self, rgb, det_label, modal_x):
        # TODO: if resize for fusion and detect train
        # resize batch images to same;or not resize in batch 1
        if self.image_resize:
            sh, sw = self.image_resize
            rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
            modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
        # 归一化0-255->0-1
        rgb = rgb.astype(np.float64) / 255.0
        modal_x = modal_x.astype(np.float64) / 255.0
        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)
        # Mask恒为1
        mask = (np.ones_like(rgb)).astype(np.float32)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        return rgb, det_label, modal_x, mask

    def _preprocess_val(self, rgb, det_label, modal_x):
        # TODO:delete;fusion with evey size;detect need same with train resize better?????
        if self.image_resize:
            sh, sw = self.image_resize
            rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
            modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
        # 归一化0-255->0-1
        rgb = rgb.astype(np.float64) / 255.0
        modal_x = modal_x.astype(np.float64) / 255.0
        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
        return rgb, det_label, modal_x

    @staticmethod
    def _read_detect_label(det_label_path):
        target = np.loadtxt(str(det_label_path), dtype=np.float32)
        # [n*[cls, cx, cy, w, h]] n是gt框个数
        labels = torch.from_numpy(target).view(-1, 5)
        # detr 网络输入cx cy w h;yolo网络输入cx cy w h
        boxes = labels[:, 1:].to(dtype=torch.float32)  # n cx cy w h
        # detr:[n*cls] yolo:[n cls];
        # classes = (labels[:, 0:1].to(dtype=torch.int64)).view(-1)  # [n*cls]
        classes = labels[:, 0:1].to(dtype=torch.int64)  # [n cls]

        detect_label = {"boxes": boxes, "labels": classes}
        return detect_label

    @staticmethod
    def collate_fn_yolo(batch):
        # only for yolo train
        # batch=[dict1,dict2,dict3,...,dictn]
        # det_label: dict{};
        #     'ori_shape': Tuple((768,1024),...*16)
        #     'resized_shape': Tuple((640,640),...*16)
        #     'img': Tensor[bs,3,h,w]
        #     'cls': Tensor[num_objects,1] need
        #     'bboxes': Tensor[num_objects,4] need
        #     'batch_idx': Tensor[num_objects]=tensor[0,0,0,0,0,1,1,2,2,2,2,2,....] 每个gt框对应的图像id索引 need

        train_val_len = len(batch[0])
        data = []
        det_label = {'cls': [], 'bboxes': [], 'batch_idx': []}
        modal_x = []
        if train_val_len == 5:
            Mask = []
        fn = []
        for idx, sample in enumerate(batch):
            data.append(sample["data"])
            modal_x.append(sample["modal_x"])
            if train_val_len == 5:
                Mask.append(sample["Mask"])
            fn.append(sample["fn"])
            # detect label;sample["det_label"] ["labels"]/["boxes"]
            det_label['cls'] += (sample["det_label"]["labels"]).tolist()
            det_label['bboxes'] += (sample["det_label"]["boxes"]).tolist()
            det_label['batch_idx'] += [idx for _ in range(len(sample["det_label"]["labels"]))]
        data = torch.stack(data, 0)
        modal_x = torch.stack(modal_x, 0)
        det_label['cls'] = torch.tensor(det_label['cls'], dtype=torch.int64)
        det_label['bboxes'] = torch.tensor(det_label['bboxes'], dtype=torch.float32)
        det_label['batch_idx'] = torch.tensor(det_label['batch_idx'], dtype=torch.int64)
        if train_val_len == 5:
            Mask = torch.stack(Mask, 0)
        if train_val_len == 5:
            return dict(data=data, det_label=det_label, modal_x=modal_x, Mask=Mask, fn=fn)
        else:
            return dict(data=data, det_label=det_label, modal_x=modal_x, fn=fn)

    @staticmethod
    def collate_fn_detr(batch):
        # for detr train;and detr/yolo eval
        # batch=[dict1,dict2,dict3,...,dictn]
        # det_lable->list
        train_val_len = len(batch[0])
        data = []
        det_label = []
        modal_x = []
        if train_val_len == 5:
            Mask = []
        fn = []
        for sample in batch:
            data.append(sample["data"])
            det_label.append(sample["det_label"])
            modal_x.append(sample["modal_x"])
            if train_val_len == 5:
                Mask.append(sample["Mask"])
            fn.append(sample["fn"])
        data = torch.stack(data, 0)
        modal_x = torch.stack(modal_x, 0)
        if train_val_len == 5:
            Mask = torch.stack(Mask, 0)
        if train_val_len == 5:
            return dict(data=data, det_label=det_label, modal_x=modal_x, Mask=Mask, fn=fn)
        else:
            return dict(data=data, det_label=det_label, modal_x=modal_x, fn=fn)


class SegDataset(data.Dataset):
    """
    dataset: contain seg infrared and visible dataset,construct like this
        seg root:
            vis
            inf
            labels(seg,mask .png/jpg/..)
            train.txt(with suffix)
            val.txt
    Args:
        setting(dict): seg_root--Absolute path to root;x_single_channel--infrared channel true for 1 else 3
        split_name(str): train/val
        is_preprocess(bool): if train/val data preprocess
    """

    def __init__(self, setting, split_name, is_preprocess=True):
        super(SegDataset, self).__init__()
        self.is_preprocess = is_preprocess
        self._split_name = split_name
        self._rgb_path = "vis"
        self._x_path = "inf"
        self._label_path = "labels"
        self._seg_root = setting['seg_root']
        self._x_single_channel = setting['x_single_channel']
        self._seg_file_names = self._get_file_names(self._seg_root, self._split_name)
        self.image_resize = setting['image_resize']

    def __len__(self):
        return len(self._seg_file_names)

    def __getitem__(self, index):
        image_name = self._seg_file_names[index]  # file.png with suffix
        rgb_path = os.path.join(self._seg_root, self._rgb_path, image_name)
        x_path = os.path.join(self._seg_root, self._x_path, image_name)
        seg_label_path = os.path.join(self._seg_root, self._label_path, image_name)

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, "RGB")

        if self._x_single_channel:
            x = self._open_image(x_path, "Gray")
            x = cv2.merge([x, x, x])
        else:
            x = self._open_image(x_path, "RGB")

        seg_label = self._open_image(seg_label_path, "Gray", dtype=np.uint8)
        # seg_label = self._seg_transform(seg_label) # background=0
        # seg_label = np.zeros((rgb.shape[0], rgb.shape[1]))

        if self.is_preprocess:
            if self._split_name == 'train':
                rgb, seg_label, x, Mask = self._preprocess_train(rgb, seg_label, x)
            elif self._split_name == 'val':
                rgb, seg_label, x = self._preprocess_val(rgb, seg_label, x)

        if self.is_preprocess and self._split_name == 'train':
            output_dict = dict(data=rgb, seg_label=seg_label, modal_x=x, Mask=Mask, fn=str(image_name))
        else:
            output_dict = dict(data=rgb, seg_label=seg_label, modal_x=x, fn=str(image_name))
        return output_dict

    @staticmethod
    def _get_file_names(data_root, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(data_root, split_name + ".txt")

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode="Gray", dtype=np.float32):
        if mode == "RGB":
            img = np.array(cv2.imread(filepath, cv2.IMREAD_COLOR), dtype=dtype)
            img = img[:, :, [2, 1, 0]]  # BGR to RGB
        elif mode == "Gray":
            img = np.array(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), dtype=dtype)
        else:
            img = np.array(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), dtype=dtype)

        return img

    @staticmethod
    def _seg_transform(seg_label):
        return seg_label - 1

    def _preprocess_train(self, rgb, seg_label, modal_x):
        # this can do some enhancement
        if self.image_resize:
            sh, sw = self.image_resize
            rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
            modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
            seg_label = cv2.resize(seg_label, (sw, sh), interpolation=cv2.INTER_NEAREST)
        # 归一化0-255->0-1
        rgb = rgb.astype(np.float64) / 255.0
        modal_x = modal_x.astype(np.float64) / 255.0
        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)
        # Mask恒为1
        mask = (np.ones_like(rgb)).astype(np.float32)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
        seg_label = torch.from_numpy(np.ascontiguousarray(seg_label)).long()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        return rgb, seg_label, modal_x, mask

    def _preprocess_val(self, rgb, seg_label, modal_x):
        if self.image_resize:
            sh, sw = self.image_resize
            rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
            modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
            seg_label = cv2.resize(seg_label, (sw, sh), interpolation=cv2.INTER_NEAREST)
        # 归一化0-255->0-1
        rgb = rgb.astype(np.float64) / 255.0
        modal_x = modal_x.astype(np.float64) / 255.0
        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
        seg_label = torch.from_numpy(np.ascontiguousarray(seg_label)).long()
        return rgb, seg_label, modal_x

    @classmethod
    def get_class_colors(cls):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 9
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(1, N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    @classmethod
    def get_class_colors_msrs(cls):
        unlabelled = [0, 0, 0]
        car = [64, 0, 128]
        person = [64, 64, 0]
        bike = [0, 128, 192]
        curve = [0, 0, 192]
        car_stop = [128, 128, 0]
        guardrail = [64, 64, 128]
        color_cone = [192, 128, 128]
        bump = [192, 64, 0]
        palette = np.array(
            [
                unlabelled,
                car,
                person,
                bike,
                curve,
                car_stop,
                guardrail,
                color_cone,
                bump,
            ]
        )
        return palette


if __name__ == '__main__':
    pass
