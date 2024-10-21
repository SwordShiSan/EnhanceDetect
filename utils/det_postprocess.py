import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import ops

from models.Detect_head import DFL
from utils import box_ops
from utils.box_ops import make_anchors, dist2bbox


class DetrPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
                     0 pred_logits 分类头输出[bs, 100, 92(类别数)]
                     1 pred_boxes 回归头输出[bs, 100, 4]
                     2 aux_outputs list: 5  前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)] 和 5个pred_boxes[bs, 100, 4]
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # out_logits：[bs, 100, 92(类别数)]
        # out_bbox：[bs, 100, 4]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # [bs, 100, 92]  对每个预测框的类别概率取softmax
        prob = F.softmax(out_logits, -1)
        # prob[..., :-1]: [bs, 100, 92] -> [bs, 100, 91]  删除背景
        # .max(-1): scores=[bs, 100]  100个预测框属于最大概率类别的概率
        #           labels=[bs, 100]  100个预测框的类别
        scores, labels = prob[..., :-1].max(-1)

        # cxcywh to xyxy  format   [bs, 100, 4]
        # 送入coco_evaluator的格式是x_min y_min x_max y_max;coco_evaluator会将其转为x_min y_min w h
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates  bs张图片的宽和高
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cuda(non_blocking=True)
        boxes = boxes * scale_fct[:, None, :]  # 归一化坐标 -> 绝对位置坐标(相对于原图的坐标)  [bs, 100, 4]

        # scores = Tensor[bs,100,]  这张图片预测的100个预测框概率分数
        # labels = Tensor[bs,100,]  这张图片预测的100个预测框所属类别idx
        # boxes = Tensor[bs,100, 4] 这张图片预测的100个预测框的绝对位置坐标(相对这张图片的原图大小的坐标)
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        # list: bs    每个元素batch都是一个dict  包括'scores'  'labels'  'boxes'三个字段
        return results


class YoloPostProcess(nn.Module):
    def __init__(self, nc=6):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.stride = torch.tensor([8, 16, 32])  # 特征图相对输入网络图像尺寸的放缩尺度
        self.device = 'cuda:0'

        self.dfl = DFL(self.reg_max).to(self.device) if self.reg_max > 1 else nn.Identity()

    @torch.no_grad()
    def forward(self, x, target_sizes, input_size=(480, 640), coco_evaluator=True, conf_thres=0.1, iou_thres=0.5):
        # x: 3 bs c h w
        # target_sizes: [tensor(h,w),...]
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # bs 4*16+6 8400
        # 2 8400;1 8400
        self.anchors, self.strides = (x.transpose(0, 1).to(self.device) for x in make_anchors(x, self.stride, 0.5))
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # bs 64 8400,bs 6 8400
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # bs 4 8400
        # bs 4+6 8400;归化到了原输入图像480*640,cx cy w h,类别经过激活函数输出概率
        preds = torch.cat((dbox, cls.sigmoid()), 1)

        # 传入cx cy w h格式,(batch_size, num_classes + 4 + num_masks, num_boxes)
        # 返回list[Tensor[num_boxes, 6 + num_masks]*bs];6=(x1, y1, x2, y2, confidence, class)
        # 0.5 0.7
        preds = ops.non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

        if not isinstance(target_sizes, list):  # input images are a torch.Tensor, not a list
            target_sizes = ops.convert_torch2numpy_batch(target_sizes)

        # 结果放缩到原图像ori_size
        results = []
        for pred, orig_size in zip(preds, target_sizes):
            pred[:, :4] = ops.scale_boxes(input_size, pred[:, :4], (int(orig_size[0]), int(orig_size[1])))

        # preds: [bs num_boxes 6],(x1, y1, x2, y2, confidence, class),
        # 送入coco_evaluator的格式是x_min y_min x_max y_max;coco_evaluator会将其转为x_min y_min w h
        if coco_evaluator:
            results = [{'scores': pred[:, 4], 'labels': pred[:, 5], 'boxes': pred[:, :4]} for pred in preds]
            return results
        return results

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)
