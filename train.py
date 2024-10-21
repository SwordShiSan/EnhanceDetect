import json
import os
import random
import sys
import time
import argparse

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from dataloader.RGBXDataset import SegDataset
from dataloader.dataloader import build_detect_loader, build_seg_loader
from models.model import MRFS, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.loss_utils import MakeLoss

from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, reduce_value, ensure_dir, get_p_according_train_mode
from utils.metric import hist_info, compute_metric
from utils.det_postprocess import DetrPostProcess, YoloPostProcess
from utils.coco_eval import build_coco_evaluator
from tensorboardX import SummaryWriter

# dataset_name = "MFNet"
dataset_name = "M3FD"
# dataset_name = "MSRS"
parser = argparse.ArgumentParser()
parser.add_argument('--continue_fpath', default="", type=str, help='Description of your argument')
parser.add_argument('--seed', default=12345, type=int, help='set seed everything')
# Dataset Config
parser.add_argument('--detect_root', default="/mnt/nas/lailihao/M3FD/", type=str,
                    help='absolute path of the detect root')
parser.add_argument('--seg_root', default="/mnt/nas/lailihao/MSRS/", type=str, help='absolute path of the seg root')
parser.add_argument('--train_size', default='', type=list, help='resize image when train if has value;[480, 640]')
parser.add_argument('--eval_size', default='', type=list, help='resize image when train if has value;[480, 640]')
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

parser.add_argument('--background', default=255, type=int,
                    help='the ignore label index of segmentation, typically need not to ignore')

parser.add_argument('--dataset_path', default="/mnt/nas/lailihao/M3FD/", type=str, help='/mnt/nas/lailihao/M3FD/')
parser.add_argument('--rgb_folder', default="Vis", type=str, help='folder for visible light images')
parser.add_argument('--rgb_format', default=".png", type=str, help='the load format for visible light images')
parser.add_argument('--x_folder', default="Inf", type=str, help='folder for thermal imaging images')
parser.add_argument('--x_format', default=".png", type=str, help='the load format for thermal imaging images')

parser.add_argument('--seg_label_folder', default="seg_label", type=str, help='folder for segmentation label image')
parser.add_argument('--seg_label_format', default=".png", type=str, help='the load format for segmentation label image')
parser.add_argument('--det_label_folder', default="labels", type=str, help='folder for detection label image')
parser.add_argument('--det_label_format', default=".txt", type=str, help='the load format for detection label txt')
parser.add_argument('--gt_transform', default=False, type=bool, help='')

parser.add_argument('--num_train_imgs', default=3400, type=int, help='60/3400')
parser.add_argument('--num_eval_imgs', default=805, type=int, help='20/805')
parser.add_argument('--image_height', default=480, type=int, help='the height of image size to train')
parser.add_argument('--image_width', default=640, type=int, help='the width of image size to train')

# Network Config
backbone = "mit_b4"
parser.add_argument('--backbone', default=backbone, type=str, help='the backbone network to load')
parser.add_argument('--pretrained_backbone', default=None, type=str,
                    help='corresponding to backbone;/mnt/nas/lailihao/Projects/MRFS/MFNet/checkpoints/log_M3FD_mit_b4/pretrain/f_s_epoch-30.pth')
parser.add_argument('--decoder_embed_dim', default=512, type=int, help='')
# Train Config
parser.add_argument('--optimizer', default="AdamW", type=str, help='optimizer')
parser.add_argument('--lr', default=6e-5, type=float, help='learning rate')
parser.add_argument('--lr_power', default=0.9, type=float, help='')
parser.add_argument('--momentum', default=0.9, type=float, help='')
parser.add_argument('--weight_decay', default=0.01, type=float, help='')
parser.add_argument('--batch_size', default=1, type=int, help='')
parser.add_argument('--nepochs', default=100, type=int, help='loop forever, stop manually')
parser.add_argument('--warm_up_epoch', default=10, type=int, help='')
parser.add_argument('--num_workers', default=4, type=int, help='16')
parser.add_argument('--train_scale_array', default=[0.5, 0.75, 1, 1.25, 1.5, 1.75], type=list, help='')
parser.add_argument('--fix_bias', default=0.01, type=bool, help='')
parser.add_argument('--bn_eps', default=0.01, type=float, help='')
parser.add_argument('--bn_momentum', default=0.01, type=float, help='')
# Val Config
parser.add_argument('--is_detect_eval', default=True, type=bool, help='if detect eval metric')
parser.add_argument('--is_seg_eval', default=False, type=bool, help='if seg eval metric')
# Store Config
parser.add_argument('--checkpoint_start_epoch', default=1, type=int, help='start to save checkpoint')
parser.add_argument('--checkpoint_step', default=10, type=int, help='save checkpoint per step')
# Logger Config
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
log_dir = f"./checkpoints/log_{dataset_name}_{backbone}/{exp_time}"
parser.add_argument('--log_dir', default=log_dir, type=str, help='where to put train log info')
parser.add_argument('--tb_dir', default=os.path.join(log_dir, "tb"), type=str, help='where to put tensorboard info')
parser.add_argument('--checkpoint_dir', default=os.path.join(log_dir, "weights"), type=str,
                    help='where to put checkpoint')
parser.add_argument('--log_train_fusion_result', default=os.path.join(log_dir, "train_result", "fusion_result"),
                    type=str, help='')
parser.add_argument('--fusion_log_file', default=os.path.join(log_dir, "train_result", "fusion_result", f"fusion.log"),
                    type=str, help='')
parser.add_argument('--log_train_detect_result', default=os.path.join(log_dir, "train_result", "detect_result"),
                    type=str, help='')
parser.add_argument('--detect_log_file', default=os.path.join(log_dir, "train_result", "detect_result", f"map.log"),
                    type=str, help='')
parser.add_argument('--log_train_seg_result', default=os.path.join(log_dir, "train_result", "seg_result"), type=str,
                    help='')
parser.add_argument('--seg_log_file', default=os.path.join(log_dir, "train_result", "seg_result", f"miou.log"),
                    type=str, help='')
# Train Mode
parser.add_argument('--train_mode', default=4, type=int,
                    help='1-train fusion;2-train detect;3-train seg;4-train fusion and detect;5-train fusion and seg;6-train fusion detect+seg')
# Some train description
parser.add_argument('--train_description',
                    default="train detect and fusion;ori_size;no contrast;use before 3 feature for fusion;0.1 * fusion_loss_total",
                    type=str,
                    help='set some description of train setting')

# TODO: record with file
logger = get_logger()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    # mkdirs log dirs if not exist
    ensure_dir(args.log_dir)
    ensure_dir(args.tb_dir)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.log_train_fusion_result)
    ensure_dir(args.log_train_detect_result)
    ensure_dir(args.log_train_seg_result)

    # 将训练参数写入文件
    with open(os.path.join(args.log_dir, "args.txt"), 'w') as f:
        # 遍历args对象的属性
        for attr_name, attr_value in vars(args).items():
            # 将参数名和参数值转换为字符串，并写入文件，每个参数占据一行
            f.write(f'{attr_name}: {attr_value}\n')

    cudnn.benchmark = True
    seed = args.seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    detect_train_loader, detect_train_sample = build_detect_loader(args, engine, "train")
    detect_val_loader, detect_val_sample = build_detect_loader(args, engine, "val")
    seg_train_loader, seg_train_sample = build_seg_loader(args, engine, "train")
    seg_val_loader, seg_val_sample = build_seg_loader(args, engine, "val")
    # seg_train_loader, seg_train_sample, seg_val_loader, seg_val_sample = None, None, None, None

    if args.train_mode == 1:
        # M3FD or MSRS
        niters_per_epoch = len(detect_train_loader)
        # niters_per_epoch = len(seg_train_loader)
    elif args.train_mode == 2 or args.train_mode == 4:
        niters_per_epoch = len(detect_train_loader)
    elif args.train_mode == 3 or args.train_mode == 5:
        niters_per_epoch = len(seg_train_loader)
    elif args.train_mode == 6:
        niters_per_epoch = len(detect_train_loader) + len(seg_train_loader)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        # tb_dir = args.tb_dir + '/{}'.format(exp_time)
        tb_dir = args.tb_dir
        tb = SummaryWriter(log_dir=tb_dir)

    # config network and criterion
    criterion = MakeLoss(args.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = MRFS(cfg=args, criterion=criterion, norm_layer=BatchNorm2d)

    base_lr = args.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = args.nepochs * niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, args.lr_power, total_iteration, niters_per_epoch * args.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    engine.register_state(model=model, optimizer=optimizer)  # just record model,opt,epochs,niters
    if engine.continue_state_object:
        engine.restore_checkpoint()

    if args.is_detect_eval:
        # 1.for detr/yolo eval
        # det_postprocessor = DetrPostProcess()
        det_postprocessor = YoloPostProcess(nc=6)
        iou_types = ('bbox',)
        coco_evaluator = build_coco_evaluator(detect_val_loader, iou_types)

    if args.is_seg_eval:
        best_seg_mIoU = 0.

    gradient_accumulation_steps = 1

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')

    for epoch in range(engine.state.epoch, args.nepochs + 1):
        if engine.distributed:
            detect_train_sample.set_epoch(epoch)
            seg_train_sample.set_epoch(epoch)
        detect_data_iter = iter(detect_train_loader)
        seg_data_iter = iter(seg_train_loader)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        # TODO: serval steps loss sum optimize if ori_size
        sum_backward_loss = 0
        sum_loss = 0
        sum_loss_seg = 0
        sum_loss_det = 0
        sum_loss_fus = 0

        for idx in pbar:

            engine.update_iteration(epoch, idx)

            # TODO: choose detect and seg
            p = get_p_according_train_mode(args.train_mode)
            if 0 <= p < 0.5:
                cur_batch_task_iter = detect_data_iter
                minibatch = next(cur_batch_task_iter)
                det_labels = minibatch['det_label']
                seg_labels = None
            else:
                cur_batch_task_iter = seg_data_iter
                minibatch = next(cur_batch_task_iter)
                seg_labels = minibatch['seg_label']
                det_labels = None

            imgs = minibatch['data']
            modal_xs = minibatch['modal_x']
            Mask = minibatch['Mask']
            name = minibatch['fn']
            # text_line = minibatch['text']

            imgs = imgs.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)
            Mask = Mask.cuda(non_blocking=True)
            # text = clip.tokenize(text_line).cuda(non_blocking=True)
            if seg_labels is not None:
                seg_labels = seg_labels.cuda(non_blocking=True)
            if det_labels is not None:
                # 1.for detr use
                # for det_label in det_labels:
                #     det_label['boxes'] = det_label['boxes'].cuda(non_blocking=True)
                #     det_label['labels'] = det_label['labels'].cuda(non_blocking=True)
                # 2.for yolo use
                det_labels['cls'] = det_labels['cls'].cuda(non_blocking=True)
                det_labels['bboxes'] = det_labels['bboxes'].cuda(non_blocking=True)
                det_labels['batch_idx'] = det_labels['batch_idx'].cuda(non_blocking=True)

            loss, seg_loss, det_loss, fus_loss = model.forward_loss(imgs, modal_xs, Mask, seg_labels, det_labels,
                                                                    args.train_mode)
            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                semantic_loss = all_reduce_tensor(seg_loss, world_size=engine.world_size)
                detection_loss = all_reduce_tensor(det_loss, world_size=engine.world_size)
                fusion_loss = all_reduce_tensor(fus_loss, world_size=engine.world_size)

            # loss = loss / gradient_accumulation_steps
            # loss.backward()
            # if (idx + 1) % gradient_accumulation_steps == 0:
            #     # 根据新的梯度更新网络参数
            #     optimizer.step()
            #     # 清空以往梯度，通过下面反向传播重新计算梯度
            #     optimizer.zero_grad()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            # console pbar print train loss each iter
            if engine.distributed:
                sum_loss += reduce_loss.item()
                sum_loss_seg += semantic_loss.item()
                sum_loss_det += detection_loss.item()
                sum_loss_fus += fusion_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, args.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1))) \
                            + ' loss_seg=%.4f total_loss_seg=%.4f' % (semantic_loss.item(), (sum_loss_seg / (idx + 1))) \
                            + ' loss_det=%.4f total_loss_det=%.4f' % (det_loss.item(), (sum_loss_det / (idx + 1))) \
                            + ' loss_fusion=%.4f total_loss_fusion=%.4f' % (
                                fusion_loss.item(), (sum_loss_fus / (idx + 1)))
            else:
                sum_loss += loss.item()
                sum_loss_seg += seg_loss.item()
                sum_loss_det += det_loss.item()
                sum_loss_fus += fus_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, args.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (loss.item(), (sum_loss / (idx + 1))) \
                            + ' loss_seg=%.4f total_loss_seg=%.4f' % (seg_loss.item(), (sum_loss_seg / (idx + 1))) \
                            + ' loss_det=%.4f total_loss_det=%.4f' % (det_loss.item(), (sum_loss_det / (idx + 1))) \
                            + ' loss_fusion=%.4f total_loss_fusion=%.4f' % (fus_loss.item(), (sum_loss_fus / (idx + 1)))

            del loss, seg_loss, det_loss, fus_loss
            pbar.set_description(print_str, refresh=False)

        # tensorboard record each epoch
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            tb.add_scalar('fusion_loss', sum_loss_fus / len(pbar), epoch)
            tb.add_scalar('detect_loss', sum_loss_det / len(pbar), epoch)
            tb.add_scalar('semantic_loss', sum_loss_seg / len(pbar), epoch)

        # save checkpoint,and val seg and det metric;if epoch achieve specific stages
        if (epoch >= args.checkpoint_start_epoch) and (epoch % args.checkpoint_step == 0) or (epoch == args.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(args.checkpoint_dir)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(args.checkpoint_dir)

            # val,do det and seg metric
            if engine.distributed and (engine.local_rank == 0) or (not engine.distributed):
                logger.info('########Validation########')

            # 1.det metric
            if args.is_detect_eval:
                if engine.distributed:
                    detect_val_sample.set_epoch(epoch)
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(detect_val_loader)), file=sys.stdout, bar_format=bar_format)
                detect_data_iter = iter(detect_val_loader)
                if coco_evaluator is not None:
                    coco_evaluator.reset()
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
                            Fus_img = Fus_img.permute(1, 2, 0)
                            # save colored result
                            result_img = Image.fromarray(Fus_img.cpu().numpy().astype(np.uint8), mode='RGB')
                            result_img.save(os.path.join(args.log_train_fusion_result, name[0]))
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
                            draw_img = cv2.cvtColor(Fus_img.cpu().numpy().astype(np.uint8),
                                                    cv2.COLOR_RGB2BGR)  # H W C(BGR)
                            ori_size = (int(det_labels[0]["orig_size"][1]), int(det_labels[0]["orig_size"][0]))
                            draw_img = cv2.resize(draw_img, ori_size, interpolation=cv2.INTER_LINEAR)  # 放缩到原图像,标签也是原图像
                            for s, l, b in zip(result["scores"], result["labels"], result["boxes"]):
                                cv2.rectangle(draw_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                                              color=box_color, thickness=2)
                                show_label = args.detect_classes[int(l)] + ',' + str(round(float(s), 2))
                                cv2.putText(draw_img, show_label, (int(b[0]), int(b[1]) - 3),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
                            cv2.imwrite(str(os.path.join(args.log_train_detect_result, name[0])), draw_img)

                if coco_evaluator is not None:
                    coco_evaluator.synchronize_between_processes()
                    coco_evaluator.accumulate()
                    coco_evaluator.summarize()
                    # 写入文件中
                    open_mode = 'a' if os.path.exists(args.detect_log_file) else 'w'
                    with open(args.detect_log_file, open_mode) as f:
                        # 保存原始的 sys.stdout
                        original_stdout = sys.stdout
                        try:
                            # 将 sys.stdout 重定向到文件
                            sys.stdout = f
                            print('Epoch {}/{}'.format(epoch, args.nepochs))
                            coco_evaluator.summarize()
                        finally:
                            # 恢复 sys.stdout 到原始值
                            sys.stdout = original_stdout

            # 2.seg metric
            if args.is_seg_eval:
                seg_all_result = []
                if engine.distributed:
                    seg_val_sample.set_epoch(epoch)
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(seg_val_loader)), file=sys.stdout, bar_format=bar_format)
                seg_data_iter = iter(seg_val_loader)

                for idx in pbar:
                    # bs=1
                    print_str = 'validation seg{}/{}'.format(idx + 1, len(seg_val_loader))
                    pbar.set_description(print_str, refresh=False)

                    data = next(seg_data_iter)
                    img = data['data']
                    seg_labels = data['seg_label']
                    modal_x = data['modal_x']
                    name = data['fn']

                    img = img.cuda(non_blocking=True)
                    seg_labels = seg_labels.cuda(non_blocking=True)
                    modal_x = modal_x.cuda(non_blocking=True)

                    with torch.no_grad():
                        seg_score, det_output, Fus_img = model(img, modal_x)
                        if idx % int(len(seg_val_loader) / 20) == 0:
                            # bs=1
                            Fus_img = Fus_img[0]
                            Fus_img = (Fus_img - Fus_img.min()) / (Fus_img.max() - Fus_img.min()) * 255.0
                            Fus_img = Fus_img.permute(1, 2, 0)
                            # save colored result
                            result_img = Image.fromarray(Fus_img.cpu().numpy().astype(np.uint8), mode='RGB')
                            result_img.save(os.path.join(args.log_train_fusion_result, name[0]))
                        if seg_labels is not None:
                            # bs=1
                            seg_score = torch.exp(seg_score[0])
                            seg_score = seg_score.permute(1, 2, 0)  # H W C
                            # seg_processed_pred = cv2.resize(seg_score.cpu().numpy(),
                            #                                 (args.eval_size[1], args.eval_size[0]),
                            #                                 interpolation=cv2.INTER_LINEAR)
                            seg_processed_pred = seg_score.cpu().numpy()  # H W C
                            seg_labels = seg_labels.squeeze().cpu().numpy()  # H W
                            seg_pred = seg_processed_pred.argmax(2)  # H W,代表分割结果,像素值是索引值代表类别;0-9
                            seg_result = hist_info(args.seg_num_classes, seg_pred, seg_labels)[0]
                            seg_all_result.append(seg_result)  # [(confusionMatrix),...]
                            if idx % int(len(seg_val_loader) / 20) == 0:
                                palette = SegDataset.get_class_colors_msrs()
                                temp = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3))
                                temp_label = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3))
                                for cid in range(args.seg_num_classes):
                                    temp[seg_pred == cid] = palette[cid]
                                    temp_label[seg_labels == cid] = palette[cid]
                                cv2.imwrite(str(os.path.join(args.log_train_seg_result, name[0])), temp)
                                cv2.imwrite(str(os.path.join(args.log_train_seg_result,
                                                             name[0][:-4] + "_label" + name[0][-4:])), temp_label)

                seg_iou, seg_mIoU = compute_metric(seg_all_result, args.seg_num_classes)
                seg_iou, seg_mIoU = seg_iou if not engine.distributed else reduce_value(seg_iou, average=True), \
                    seg_mIoU if not engine.distributed else reduce_value(seg_mIoU, average=True)

                if engine.distributed and (engine.local_rank == 0) or (not engine.distributed):
                    result_line = [f"{args.seg_classes[i]:8s}: \t {seg_iou[i] * 100:.3f}% \n" for i in
                                   range(args.seg_num_classes)]
                    result_line.append(f"mean seg_iou: \t {seg_mIoU[0] * 100:.3f}% \n")
                    open_mode = 'a' if os.path.exists(args.seg_log_file) else 'w'
                    results = open(args.seg_log_file, open_mode)
                    results.write(f"##epoch:{epoch:4d} " + "#" * 67 + "\n")
                    print("#" * 80 + "\n")
                    for line in result_line:
                        print(line)
                        results.write(line)
                    results.write("#" * 80 + "\n")
                    results.flush()
                    results.close()
                    if seg_mIoU[0] > best_seg_mIoU:
                        best_seg_mIoU = seg_mIoU[0]
                        engine.save_checkpoint(os.path.join(args.checkpoint_dir, "best_seg_mIoU.pth"))
