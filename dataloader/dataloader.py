import os
import cv2
import torch
import numpy as np
from torch.utils import data
import random

from dataloader.RGBXDataset import DetectDataset, SegDataset
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize


def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x


def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale


def collate_fn_detr(batch):
    # batch=[dict1,dict2,dict3,...,dictn]
    train_val_len = len(batch[0])
    # data = rgb, seg_label = gt, det_label = target, modal_x = x, Mask = Mask, fn = str(item_name),n = len(self._file_names)
    # data = rgb, seg_label = gt, det_label = target, modal_x = x, fn = str(item_name),n = len(self._file_names)
    data = []
    seg_label = []
    det_label = []
    modal_x = []
    Mask = []
    fn = []
    n = []
    for sample in batch:
        data.append(sample["data"])
        seg_label.append(sample["seg_label"])
        det_label.append(sample["det_label"])
        modal_x.append(sample["modal_x"])
        if train_val_len == 7:
            Mask.append(sample["Mask"])
        fn.append(sample["fn"])
        n.append(sample["n"])
    data = torch.stack(data, 0)
    seg_label = torch.stack(seg_label, 0)
    modal_x = torch.stack(modal_x, 0)
    if train_val_len == 7:
        Mask = torch.stack(Mask, 0)
    # fn = torch.stack(fn, 0)
    # n = torch.stack(n, 0)
    if train_val_len == 7:
        return dict(data=data, seg_label=seg_label, det_label=det_label, modal_x=modal_x, Mask=Mask, fn=fn, n=n)
    else:
        return dict(data=data, seg_label=seg_label, det_label=det_label, modal_x=modal_x, fn=fn, n=n)


def collate_fn_yolo(batch):
    # batch=[dict1,dict2,dict3,...,dictn]
    # data = rgb, seg_label = gt, det_label = target, modal_x = x, Mask = Mask, fn = str(item_name),n = len(self._file_names)
    # data = rgb, seg_label = gt, det_label = target, modal_x = x, fn = str(item_name),n = len(self._file_names)

    # det_label: dict{};
    #     'ori_shape': Tuple((768,1024),...*16)
    #     'resized_shape': Tuple((640,640),...*16)
    #     'img': Tensor[bs,3,h,w]
    #     'cls': Tensor[num_objects,1] need
    #     'bboxes': Tensor[num_objects,4] need
    #     'batch_idx': Tensor[num_objects]=tensor[0,0,0,0,0,1,1,2,2,2,2,2,....] 每个gt框对应的图像id索引 need

    train_val_len = len(batch[0])

    data = []
    seg_label = []
    det_label = {'cls': [], 'bboxes': [], 'batch_idx': []}
    modal_x = []
    Mask = []
    fn = []
    n = []
    for idx, sample in enumerate(batch):
        data.append(sample["data"])
        seg_label.append(sample["seg_label"])
        modal_x.append(sample["modal_x"])
        if train_val_len == 7:
            Mask.append(sample["Mask"])
        fn.append(sample["fn"])
        n.append(sample["n"])
        # detect label;sample["det_label"] ["labels"]/["boxes"]
        det_label['cls'] += (sample["det_label"]["labels"]).tolist()
        det_label['bboxes'] += (sample["det_label"]["boxes"]).tolist()
        det_label['batch_idx'] += [idx for _ in range(len(sample["det_label"]["labels"]))]
    data = torch.stack(data, 0)
    seg_label = torch.stack(seg_label, 0)
    modal_x = torch.stack(modal_x, 0)
    det_label['cls'] = torch.tensor(det_label['cls'], dtype=torch.int64)
    det_label['bboxes'] = torch.tensor(det_label['bboxes'], dtype=torch.float32)
    det_label['batch_idx'] = torch.tensor(det_label['batch_idx'], dtype=torch.int64)
    if train_val_len == 7:
        Mask = torch.stack(Mask, 0)
    # fn = torch.stack(fn, 0)
    # n = torch.stack(n, 0)
    if train_val_len == 7:
        return dict(data=data, seg_label=seg_label, det_label=det_label, modal_x=modal_x, Mask=Mask, fn=fn, n=n)
    else:
        return dict(data=data, seg_label=seg_label, det_label=det_label, modal_x=modal_x, fn=fn, n=n)


class TrainPre(object):
    def __init__(self, image_height, image_width, train_scale_array):
        self.image_height = image_height
        self.image_width = image_width
        self.train_scale_array = train_scale_array

    def __call__(self, rgb, gt, modal_x):
        # 这里的数据增强检测label不太好处理,直接取消涉及检测label变动的增强

        # 随机翻转,影响检测label,取消该增强
        # rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        # 随机放缩,不影响检测label
        # if self.train_scale_array is not None:
        #     rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.train_scale_array)
        sw = 640
        sh = 480
        rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
        modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
        # 归一化0-255->0-1
        rgb = normalize(rgb)
        modal_x = normalize(modal_x)

        # 随机crop,影响检测label,取消该增强
        # # 随机crop 640x480大小,如果原图小于这个,则返回crop_pos:x=0,y=0
        # crop_size = (self.image_height, self.image_width)
        # crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)
        #
        # # Margin记录pad的尺寸,
        # p_rgb, Margin = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        # p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        # p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
        #
        # p_rgb = p_rgb.transpose(2, 0, 1)
        # p_modal_x = p_modal_x.transpose(2, 0, 1)
        #
        # # 产生Mask,用于记录pad的像素,居中的原图像
        # Mask = np.zeros(p_rgb.shape)
        # Mask[:, Margin[0]:(crop_size[0]-Margin[1]), Margin[2]:(crop_size[1]-Margin[3])] = 1.
        # return p_rgb, p_gt, p_modal_x, Mask.astype(np.float32)

        # Mask恒为1
        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)
        Mask = np.ones_like(rgb)

        return rgb, gt, modal_x, Mask.astype(np.float32)


class ValPre(object):
    def __call__(self, rgb, gt, modal_x):
        return rgb, gt, modal_x


def get_train_loader(config, engine, dataset):
    data_setting = {'rgb_root': os.path.join(config.dataset_path, config.rgb_folder),
                    'rgb_format': config.rgb_format,
                    'x_root': os.path.join(config.dataset_path, config.x_folder),
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'gt_root': os.path.join(config.dataset_path, config.seg_label_folder),
                    'gt_format': config.seg_label_format,
                    'det_label_root': os.path.join(config.dataset_path, config.det_label_folder),
                    'det_label_format': config.det_label_format,
                    'transform_gt': config.gt_transform,
                    'class_names': config.class_names,
                    'train_source': os.path.join(config.dataset_path, "train.txt"),
                    'eval_source': os.path.join(config.dataset_path, "val.txt"),
                    }
    train_preprocess = TrainPre(config.image_height,
                                config.image_width,
                                config.train_scale_array,
                                )
    file_length = (config.num_train_imgs // config.batch_size + 1) * config.batch_size
    train_dataset = dataset(data_setting, "train", train_preprocess, file_length)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   collate_fn=collate_fn_yolo)

    return train_loader, train_sampler


def get_val_loader(config, engine, dataset):
    data_setting = {'rgb_root': os.path.join(config.dataset_path, config.rgb_folder),
                    'rgb_format': config.rgb_format,
                    'x_root': os.path.join(config.dataset_path, config.x_folder),
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'gt_root': os.path.join(config.dataset_path, config.seg_label_folder),
                    'gt_format': config.seg_label_format,
                    'det_label_root': os.path.join(config.dataset_path, config.det_label_folder),
                    'det_label_format': config.det_label_format,
                    'transform_gt': config.gt_transform,
                    'class_names': config.class_names,
                    'train_source': os.path.join(config.dataset_path, "train.txt"),
                    'eval_source': os.path.join(config.dataset_path, "val.txt"),
                    }

    class ValPre(object):
        def __call__(self, rgb, gt, modal_x):
            rgb = normalize(rgb)
            modal_x = normalize(modal_x)
            # sw = 640
            # sh = 480
            # rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
            # gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
            # modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
            rgb = rgb.transpose(2, 0, 1)
            modal_x = modal_x.transpose(2, 0, 1)
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
            return rgb, gt, modal_x

    val_pre = ValPre()

    val_sampler = None
    is_shuffle = False
    batch_size = 1

    val_dataset = dataset(data_setting, 'val', val_pre)

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 shuffle=is_shuffle,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 collate_fn=collate_fn_detr)
    return val_loader, val_sampler


def build_detect_loader(config, engine, split_name):
    if split_name == "train":
        data_setting = {'detect_root': config.detect_root, 'x_single_channel': config.x_is_single_channel,
                        'image_resize': config.train_size}
    else:
        data_setting = {'detect_root': config.detect_root, 'x_single_channel': config.x_is_single_channel,
                        'image_resize': config.eval_size}
    dataset = DetectDataset(setting=data_setting, split_name=split_name, is_preprocess=True)

    sampler = None
    is_shuffle = True if split_name == "train" else False
    batch_size = config.batch_size if split_name == "train" else 1
    drop_last = True if split_name == "train" else False
    # train dataloader detr/yolo;val dataloader follow detr organized for coco_evaluator
    collate_fn = dataset.collate_fn_yolo if split_name == "train" else dataset.collate_fn_detr

    if engine.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=config.num_workers,
                             drop_last=drop_last,
                             shuffle=is_shuffle,
                             pin_memory=True,
                             sampler=sampler,
                             collate_fn=collate_fn)

    return loader, sampler


def build_seg_loader(config, engine, split_name):
    if split_name == "train":
        data_setting = {'seg_root': config.seg_root, 'x_single_channel': config.x_is_single_channel,
                        'image_resize': config.train_size}
    else:
        data_setting = {'seg_root': config.seg_root, 'x_single_channel': config.x_is_single_channel,
                        'image_resize': config.eval_size}
    dataset = SegDataset(setting=data_setting, split_name=split_name, is_preprocess=True)

    sampler = None
    is_shuffle = True if split_name == "train" else False
    batch_size = config.batch_size if split_name == "train" else 1
    drop_last = True if split_name == "train" else False

    if engine.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=config.num_workers,
                             drop_last=drop_last,
                             shuffle=is_shuffle,
                             pin_memory=True,
                             sampler=sampler)

    return loader, sampler
