import os
import argparse
import sys
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.coco_eval import build_coco_evaluator
from utils.det_postprocess import YoloPostProcess
from utils.pyt_utils import ensure_dir, parse_devices
from engine.evaluator_vision import Evaluator
from engine.logger import get_logger
from dataloader.RGBXDataset import RGBXDataset, DetectDataset, SegDataset
from models.model import MRFS

logger = get_logger()


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if model_file is None:
        return model
    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        elif 'module' in state_dict.keys():
            state_dict = state_dict['module']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=True)
    # ckpt_keys = set(state_dict.keys())
    # own_keys = set(model.state_dict().keys())
    # missing_keys = own_keys - ckpt_keys
    # unexpected_keys = ckpt_keys - own_keys

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def eval_detect_fusion(args):
    model = MRFS(cfg=args, criterion=None, norm_layer=nn.BatchNorm2d)
    load_model(model, args.checkpoint_path)
    model = model.to(args.device)

    data_setting = {'detect_root': args.detect_root, 'x_single_channel': args.x_is_single_channel,
                    'image_resize': args.eval_size}
    detect_val_dataset = DetectDataset(setting=data_setting, split_name="val", is_preprocess=True)
    detect_val_loader = DataLoader(detect_val_dataset,
                                   batch_size=1,
                                   num_workers=4,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=True,
                                   collate_fn=detect_val_dataset.collate_fn_detr)

    det_postprocessor = YoloPostProcess(nc=6)
    iou_types = ('bbox',)
    coco_evaluator = build_coco_evaluator(detect_val_loader, iou_types)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(len(detect_val_loader)), file=sys.stdout, bar_format=bar_format)
    detect_data_iter = iter(detect_val_loader)
    if coco_evaluator is not None:
        coco_evaluator.reset()

    # model.eval()  这个会影响精度?可能是因为训练的时候并没有model.eval(),导致BatchNorm
    logger.info("Start detect eval")
    # TODO: 分开测试fusion和detect;fusion用原尺寸,detect用训练尺寸
    for idx in pbar:
        # bs=1
        print_str = 'validation detection{}/{}'.format(idx + 1, len(detect_val_loader))
        pbar.set_description(print_str, refresh=False)

        data = next(detect_data_iter)
        img = data['data']
        det_labels = data['det_label']
        modal_x = data['modal_x']
        name = data['fn']

        b, c, input_h, input_w = img.shape

        img = img.cuda(non_blocking=True)
        # for det_label in det_labels:
        #     det_label['boxes'] = det_label['boxes'].cuda(non_blocking=True)
        #     det_label['labels'] = det_label['labels'].cuda(non_blocking=True)
        modal_x = modal_x.cuda(non_blocking=True)

        with torch.no_grad():
            seg_score, det_output, Fus_img = model(img, modal_x)
            if idx % int(len(detect_val_loader) / 20) == 0:
                # bs=1
                Fus_img = Fus_img[0]
                Fus_img = (Fus_img - Fus_img.min()) / (Fus_img.max() - Fus_img.min()) * 255.0
                Fus_img = Fus_img.permute(1, 2, 0)  # H W C(RGB)
                # save colored result
                result_img = Image.fromarray(Fus_img.cpu().numpy().astype(np.uint8), mode='RGB')
                result_img.save(os.path.join(args.save_path, "fusion_result", name[0]))
            if coco_evaluator is not None:
                # 1.for detr coco_evaluator
                # orig_target_sizes = torch.stack([t["orig_size"] for t in det_labels], dim=0)  # [bs]
                # results = det_postprocessor(det_output, orig_target_sizes)
                # res = {target['image_id'].item(): output for target, output in zip(det_labels, results)}

                # 2.for yolo coco_evaluator
                orig_target_sizes = [t["orig_size"] for t in det_labels]
                input_size = (input_h, input_w)
                results = det_postprocessor(det_output, orig_target_sizes, input_size=input_size,
                                            coco_evaluator=True, conf_thres=0.1, iou_thres=0.5)
                res = {target['image_id']: output for target, output in zip(det_labels, results)}

                # update results in coco_evaluator
                coco_evaluator.update(res)
            # draw rectangle in samples
            if idx % int(len(detect_val_loader) / 20) == 0:
                # save visual result
                # bs=1
                result = results[0]
                box_color = (0, 0, 255)
                draw_img = cv2.cvtColor(Fus_img.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)  # H W C(BGR)
                ori_size = (int(det_labels[0]["orig_size"][1]), int(det_labels[0]["orig_size"][0]))
                draw_img = cv2.resize(draw_img, ori_size, interpolation=cv2.INTER_LINEAR)  # 放缩到原图像,标签也是原图像
                for s, l, b in zip(result["scores"], result["labels"], result["boxes"]):
                    cv2.rectangle(draw_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                                  color=box_color, thickness=2)
                    show_label = args.detect_classes[int(l)] + ',' + str(round(float(s), 2))
                    cv2.putText(draw_img, show_label, (int(b[0]), int(b[1]) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
                cv2.imwrite(args.save_path + "detect_result/" + name[0], draw_img)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        # 获取每个类别的 AP 值
        categoryIds = coco_evaluator.coco_gt.getCatIds()
        ap_per_cat = {coco_evaluator.coco_gt.loadCats(catId)[0]['name']: float(ap) for catId, ap in
                      zip(categoryIds, coco_evaluator.coco_eval['bbox'].stats[1:])}
        # 打印每个类别的 AP 值
        for category, ap in ap_per_cat.items():
            print(f'{category}: {ap:.4f}')

        # 写入文件中
        open_mode = 'a' if args.save_path + "detect_map.txt" else 'w'
        with open(args.save_path + "detect_map.txt", open_mode) as f:
            # 保存原始的 sys.stdout
            original_stdout = sys.stdout
            try:
                # 将 sys.stdout 重定向到文件
                sys.stdout = f
                print("Epoch:200/200")
                coco_evaluator.summarize()
            finally:
                # 恢复 sys.stdout 到原始值
                sys.stdout = original_stdout


def eval_seg_fusion(args):
    # use eval_seg.py
    pass


if __name__ == "__main__":
    dataset_name = "M3FD"
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', '-p', default='./test_result/')
    # Dataset Config
    parser.add_argument('--detect_root', default="/mnt/nas/lailihao/M3FD/", type=str,
                        help='absolute path of the detect root')
    parser.add_argument('--seg_root', default="/mnt/nas/lailihao/MSRS/", type=str, help='absolute path of the seg root')
    parser.add_argument('--x_is_single_channel', default=True, type=bool,
                        help='True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input')
    parser.add_argument('--detect_num_classes', default=6, type=int, help='detect num classes')
    parser.add_argument('--detect_classes', default=['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck'],
                        type=list, help='detect object classes')
    parser.add_argument('--seg_num_classes', default=9, type=int, help='seg num classes')
    parser.add_argument('--seg_classes',
                        default=['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone',
                                 'bump'],
                        type=list, help='the class names of all classes')
    parser.add_argument('--eval_size', default=[], type=list,
                        help='resize image when train if has value;[480, 640]')
    # Network Config
    backbone = "mit_b4"
    parser.add_argument('--backbone', default=backbone, type=str, help='the backbone network to load')
    parser.add_argument('--decoder_embed_dim', default=512, type=int, help='')

    # Val Config
    parser.add_argument('--eval_crop_size', default=[480, 640], type=list, help='')
    parser.add_argument('--eval_stride_rate', default=2 / 3, type=float, help='')
    parser.add_argument('--eval_scale_array', default=[1], type=list, help='')
    parser.add_argument('--is_flip', default=False, type=bool, help='')
    log_dir = f"./checkpoints/log_{dataset_name}_{backbone}"
    parser.add_argument('--log_dir', default=log_dir, type=str, help=' ')
    parser.add_argument('--log_dir_link', default=log_dir, type=str, help='')
    parser.add_argument('--checkpoint_dir', default=os.path.join(log_dir, "weights"), type=str, help='')
    parser.add_argument('--checkpoint_path',
                        default="/mnt/nas/lailihao/Projects/MRFS/MFNet/checkpoints/log_M3FD_mit_b4/2024_10_12_11_06_54/weights/epoch-70.pth",
                        type=str, help='')

    args = parser.parse_args()

    eval_detect_fusion(args=args)
