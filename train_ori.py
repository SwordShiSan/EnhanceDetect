import os
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

from dataloader.dataloader import get_train_loader, get_val_loader
from dataloader.RGBXDataset import RGBXDataset
from models.model import MRFS, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.loss_utils import MakeLoss

from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, reduce_value
from utils.metric import hist_info, compute_metric
from utils.det_postprocess import DetrPostProcess, YoloPostProcess
from utils.coco_eval import build_coco_evaluator
from tensorboardX import SummaryWriter

dataset_name = "MFNet"
parser = argparse.ArgumentParser()
parser.add_argument('--continue_fpath', default="", type=str, help='Description of your argument')
parser.add_argument('--seed', default=12345, type=int, help='set seed everything')
# Dataset Config
parser.add_argument('--dataset_path', default="/mnt/nas/lailihao/M3FD/", type=str,
                    help='absolute path of the dataset root;/mnt/nas/lailihao/MSRS_ori/detection/;/mnt/nas/lailihao/M3FD/')
parser.add_argument('--rgb_folder', default="Vis", type=str, help='folder for visible light images')
parser.add_argument('--rgb_format', default=".png", type=str, help='the load format for visible light images')
parser.add_argument('--x_folder', default="Inf", type=str, help='folder for thermal imaging images')
parser.add_argument('--x_format', default=".png", type=str, help='the load format for thermal imaging images')
parser.add_argument('--x_is_single_channel', default=True, type=bool,
                    help='True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input')
parser.add_argument('--seg_label_folder', default="seg_label", type=str, help='folder for segmentation label image')
parser.add_argument('--seg_label_format', default=".png", type=str, help='the load format for segmentation label image')
parser.add_argument('--det_label_folder', default="labels", type=str, help='folder for detection label image')
parser.add_argument('--det_label_format', default=".txt", type=str, help='the load format for detection label txt')
parser.add_argument('--gt_transform', default=False, type=bool, help='')
parser.add_argument('--num_classes', default=9, type=int, help='')
parser.add_argument('--class_names',
                    default=['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone',
                             'bump'],
                    type=list, help='the class names of all classes')
parser.add_argument('--num_train_imgs', default=3400, type=int, help='60/3400')
parser.add_argument('--num_eval_imgs', default=805, type=int, help='20/805')
# Image Config
parser.add_argument('--image_height', default=480, type=int, help='the height of image size to train')
parser.add_argument('--image_width', default=640, type=int, help='the width of image size to train')
parser.add_argument('--background', default=255, type=int,
                    help='the ignore label index of segmentation, typically need not to ignore')
# Network Config
backbone = "mit_b4"
parser.add_argument('--backbone', default=backbone, type=str, help='the backbone network to load')
parser.add_argument('--pretrained_backbone', default=None, type=str,
                    help='corresponding to backbone ./checkpoints/pretrained/mit_b4.pth')
parser.add_argument('--decoder_embed_dim', default=512, type=int, help='')
# Train Config
parser.add_argument('--optimizer', default="AdamW", type=str, help='optimizer')
parser.add_argument('--lr', default=6e-5, type=float, help='learning rate')
parser.add_argument('--lr_power', default=0.9, type=float, help='')
parser.add_argument('--momentum', default=0.9, type=float, help='')
parser.add_argument('--weight_decay', default=0.01, type=float, help='')
parser.add_argument('--batch_size', default=4, type=int, help='')
parser.add_argument('--nepochs', default=200, type=int, help='loop forever, stop manually')
parser.add_argument('--warm_up_epoch', default=10, type=int, help='')
parser.add_argument('--num_workers', default=16, type=int, help='')
parser.add_argument('--train_scale_array', default=[0.5, 0.75, 1, 1.25, 1.5, 1.75], type=list, help='')
parser.add_argument('--fix_bias', default=0.01, type=bool, help='')
parser.add_argument('--bn_eps', default=0.01, type=float, help='')
parser.add_argument('--bn_momentum', default=0.01, type=float, help='')
# Val Config
parser.add_argument('--eval_size', default=[480, 640], type=list, help='image size')
# Store Config
parser.add_argument('--checkpoint_start_epoch', default=10, type=int, help='start to save checkpoint')
parser.add_argument('--checkpoint_step', default=10, type=int, help='save checkpoint per step')
# Logger Config
log_dir = f"./checkpoints/log_{dataset_name}_{backbone}"
parser.add_argument('--log_dir', default=log_dir, type=str, help='where to put train log info')
parser.add_argument('--tb_dir', default=os.path.join(log_dir, "tb"), type=str, help='where to put tensorboard info')
parser.add_argument('--checkpoint_dir', default=os.path.join(log_dir, "weights"), type=str,
                    help='where to put checkpoint')
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
parser.add_argument('--log_file', default=os.path.join(log_dir, f"train_{exp_time}.log"), type=str, help='')

logger = get_logger()
with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = args.seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # train dataloader detr/yolo;val dataloader follow detr organized for coco_evaluator
    train_loader, train_sampler = get_train_loader(args, engine, RGBXDataset)
    val_loader, val_sampler = get_val_loader(args, engine, RGBXDataset)

    # TODO: remove add 1
    niters_per_epoch = args.num_train_imgs // args.batch_size + 1  # 3400//4+1=851
    # niters_per_epoch = len(train_loader)+1

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = args.tb_dir + '/{}'.format(exp_time)
        tb = SummaryWriter(log_dir=tb_dir)

    # config network and criterion
    # criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    criterion = MakeLoss(args.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = MRFS(cfg=args, criterion=criterion, norm_layer=BatchNorm2d)
    # print(model)
    # from collections import OrderedDict
    #
    # state_dict=model.state_dict()
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     key = k
    #     if k.split('.')[0] == 'module':
    #         key = k[7:]
    #     new_state_dict[key] = v

    # group weight and config optimizer
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

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    # 1.for detr/yolo eval
    # det_postprocessor = DetrPostProcess()
    det_postprocessor = YoloPostProcess(nc=6)
    iou_types = ('bbox',)
    coco_evaluator = build_coco_evaluator(val_loader, iou_types)
    # 2.for yolo eval
    # coco_evaluator = None
    best_seg_mIoU = 0.

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')

    for epoch in range(engine.state.epoch, args.nepochs + 1):
        # TODO: random choose detect or seg
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        sum_loss_seg = 0
        sum_loss_det = 0
        sum_loss_fus = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            # minibatch = dataloader.next()
            minibatch = next(dataloader)
            imgs = minibatch['data']
            seg_labels = minibatch['seg_label']
            seg_labels = None
            det_labels = minibatch['det_label']
            modal_xs = minibatch['modal_x']
            Mask = minibatch['Mask']
            name = minibatch['fn']

            imgs = imgs.cuda(non_blocking=True)
            # seg_labels = seg_labels.cuda(non_blocking=True)
            # 1.for detr use
            # for det_label in det_labels:
            #     det_label['boxes'] = det_label['boxes'].cuda(non_blocking=True)
            #     det_label['labels'] = det_label['labels'].cuda(non_blocking=True)
            # 2.for yolo use
            det_labels['cls'] = det_labels['cls'].cuda(non_blocking=True)
            det_labels['bboxes'] = det_labels['bboxes'].cuda(non_blocking=True)
            det_labels['batch_idx'] = det_labels['batch_idx'].cuda(non_blocking=True)

            modal_xs = modal_xs.cuda(non_blocking=True)
            Mask = Mask.cuda(non_blocking=True)

            loss, seg_loss, det_loss, fus_loss = model.forward_loss(imgs, modal_xs, Mask, seg_labels, det_labels)
            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                semantic_loss = all_reduce_tensor(seg_loss, world_size=engine.world_size)
                fusion_loss = all_reduce_tensor(fus_loss, world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                sum_loss_seg += semantic_loss.item()
                sum_loss_det += det_loss.item()
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
                sum_loss += loss
                sum_loss_seg += seg_loss
                sum_loss_det += det_loss.item()
                sum_loss_fus += fus_loss
                print_str = 'Epoch {}/{}'.format(epoch, args.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1))) \
                            + ' loss_seg=%.4f total_loss_seg=%.4f' % (seg_loss, (sum_loss_seg / (idx + 1))) \
                            + ' loss_det=%.4f total_loss_det=%.4f' % (det_loss.item(), (sum_loss_det / (idx + 1))) \
                            + ' loss_fusion=%.4f total_loss_fusion=%.4f' % (fus_loss, (sum_loss_fus / (idx + 1)))

            del loss, seg_loss, det_loss, fus_loss
            pbar.set_description(print_str, refresh=False)

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            tb.add_scalar('fusion_loss', sum_loss_fus / len(pbar), epoch)
            tb.add_scalar('detect_loss', sum_loss_det / len(pbar), epoch)
            tb.add_scalar('semantic_loss', sum_loss_seg / len(pbar), epoch)

        if (epoch >= args.checkpoint_start_epoch) and (epoch % args.checkpoint_step == 0) or (epoch == args.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(args.checkpoint_dir)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(args.checkpoint_dir)

            ### val,do seg and det metric

            if engine.distributed and (engine.local_rank == 0) or (not engine.distributed):
                logger.info('########Validation########')
            seg_all_result = []
            if engine.distributed:
                val_sampler.set_epoch(epoch)
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(len(val_loader)), file=sys.stdout, bar_format=bar_format)
            dataloader = iter(val_loader)
            if coco_evaluator is not None:
                coco_evaluator.reset()

            for idx in pbar:
                # bs=1
                print_str = 'validation {}/{}'.format(idx + 1, len(val_loader))
                pbar.set_description(print_str, refresh=False)

                # data = dataloader.next()
                data = next(dataloader)
                img = data['data']
                seg_labels = data['seg_label']
                seg_labels = None
                det_labels = data['det_label']
                modal_x = data['modal_x']
                name = data['fn']

                img = img.cuda(non_blocking=True)
                # for det_label in det_labels:
                #     det_label['boxes'] = det_label['boxes'].cuda(non_blocking=True)
                #     det_label['labels'] = det_label['labels'].cuda(non_blocking=True)
                modal_x = modal_x.cuda(non_blocking=True)

                with torch.no_grad():
                    seg_score, det_output, Fus_img = model(img, modal_x)
                    if idx % (args.num_eval_imgs / 10) == 0:
                        # bs=1
                        Fus_img = Fus_img[0]
                        Fus_img = (Fus_img - Fus_img.min()) / (Fus_img.max() - Fus_img.min()) * 255.0
                        Fus_img = Fus_img.permute(1, 2, 0)
                        fn = name[0] + '.png'
                        # save colored result
                        result_img = Image.fromarray(Fus_img.cpu().numpy().astype(np.uint8), mode='RGB')
                        result_img.save(os.path.join(
                            "checkpoints/log_M3FD_mit_b4/2024_10_10_09_00_27/train_result/fusion_result", fn))
                    if seg_labels:
                        # bs=1
                        seg_score = torch.exp(seg_score[0])
                        seg_score = seg_score.permute(1, 2, 0)
                        seg_processed_pred = cv2.resize(seg_score.cpu().numpy(), (args.eval_size[1], args.eval_size[0]),
                                                        interpolation=cv2.INTER_LINEAR)
                        seg_pred = seg_processed_pred.argmax(2)  # H W,代表分割结果,像素值是索引值代表类别;0-9
                        seg_result = hist_info(args.num_classes, seg_pred, seg_labels.squeeze().numpy())[0]
                        seg_all_result.append(seg_result)  # [(confusionMatrix),...]
                        if idx % (args.num_eval_imgs / 10) == 0:
                            pass
                    if det_labels:
                        if coco_evaluator is not None:
                            # 1.for detr coco_evaluator
                            # orig_target_sizes = torch.stack([t["orig_size"] for t in det_labels], dim=0)  # [bs]
                            # results = det_postprocessor(det_output, orig_target_sizes)
                            # res = {target['image_id'].item(): output for target, output in zip(det_labels, results)}

                            # 2.for yolo coco_evaluator
                            orig_target_sizes = [t["orig_size"] for t in det_labels]
                            results = det_postprocessor(det_output, orig_target_sizes, input_size=(480, 640),
                                                        coco_evaluator=True)
                            res = {target['image_id']: output for target, output in zip(det_labels, results)}
                            # update results in coco_evaluator
                            coco_evaluator.update(res)
                            # print(1111111)
                            if idx % (args.num_eval_imgs / 10) == 0:
                                # save visual result
                                # bs=1
                                result = results[0]
                                box_color = (255, 0, 255)
                                draw_img = cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                                label_dict = {0: 'People', 1: 'Car', 2: 'Bus', 3: 'Lamp', 4: 'Motorcycle', 5: 'Truck'}
                                for res_box in result:
                                    cv2.rectangle(draw_img, (res_box[0], res_box[1]), (res_box[2], res_box[3]),
                                                  color=box_color, thickness=2)
                                    show_label = label_dict[int(res_box[5])] + ',' + str(round(float(res_box[4]), 2))
                                    cv2.putText(draw_img, show_label, (res_box[0], res_box[1] - 3),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1)
                                cv2.imwrite(os.path.join(
                                    "checkpoints/log_M3FD_mit_b4/2024_10_10_09_00_27/train_result/detect_result/", fn), draw_img)

            if seg_labels:
                seg_iou, seg_mIoU = compute_metric(seg_all_result, args.num_classes)
                seg_iou, seg_mIoU = seg_iou if not engine.distributed else reduce_value(seg_iou, average=True), \
                    seg_mIoU if not engine.distributed else reduce_value(seg_mIoU, average=True)

                if engine.distributed and (engine.local_rank == 0) or (not engine.distributed):
                    result_line = [f"{args.class_names[i]:8s}: \t {seg_iou[i] * 100:.3f}% \n" for i in
                                   range(args.num_classes)]
                    result_line.append(f"mean seg_iou: \t {seg_mIoU[0] * 100:.3f}% \n")

                    results = open(args.log_file, 'a')
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
            if det_labels:
                if coco_evaluator is not None:
                    coco_evaluator.synchronize_between_processes()
                    coco_evaluator.accumulate()
                    coco_evaluator.summarize()
