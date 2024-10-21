import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from utils import box_ops
from utils import matcher
from utils.box_ops import *


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat([Y, Cr, Cb], dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def Sobelxy(x):
    kernelx = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    kernely = [[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    sobelx = F.conv2d(x, weightx, padding=1)
    sobely = F.conv2d(x, weighty, padding=1)
    return sobelx, sobely


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, input_xy, output, Mask):
        input_vis, input_ir = input_xy

        Fuse = output * Mask
        YCbCr_Fuse = RGB2YCrCb(Fuse)
        Y_Fuse = YCbCr_Fuse[:, 0:1, :, :]
        Cr_Fuse = YCbCr_Fuse[:, 1:2, :, :]
        Cb_Fuse = YCbCr_Fuse[:, 2:, :, :]

        # TODO: if adjust contrast;when train seg MSRS if open?
        # R_vis = torchvision.transforms.functional.adjust_gamma(input_vis, 0.5, 1)
        # R_vis = torchvision.transforms.functional.adjust_gamma(input_vis, 1, 1)
        R_vis = input_vis
        YCbCr_R_vis = RGB2YCrCb(R_vis)
        Y_R_vis = YCbCr_R_vis[:, 0:1, :, :]
        Cr_R_vis = YCbCr_R_vis[:, 1:2, :, :]
        Cb_R_vis = YCbCr_R_vis[:, 2:, :, :]

        # R_ir = torchvision.transforms.functional.adjust_contrast(input_ir, 1.7)
        # R_ir = torchvision.transforms.functional.adjust_contrast(input_ir, 1)
        R_ir = input_ir

        Fuse_R = torch.unsqueeze(Fuse[:, 0, :, :], 1)
        Fuse_G = torch.unsqueeze(Fuse[:, 1, :, :], 1)
        Fuse_B = torch.unsqueeze(Fuse[:, 2, :, :], 1)
        Fuse_R_grad_x, Fuse_R_grad_y = Sobelxy(Fuse_R)
        Fuse_G_grad_x, Fuse_G_grad_y = Sobelxy(Fuse_G)
        Fuse_B_grad_x, Fuse_B_grad_y = Sobelxy(Fuse_B)
        Fuse_grad_x = torch.cat([Fuse_R_grad_x, Fuse_G_grad_x, Fuse_B_grad_x], 1)
        Fuse_grad_y = torch.cat([Fuse_R_grad_y, Fuse_G_grad_y, Fuse_B_grad_y], 1)

        R_VIS_R = torch.unsqueeze(R_vis[:, 0, :, :], 1)
        R_VIS_G = torch.unsqueeze(R_vis[:, 1, :, :], 1)
        R_VIS_B = torch.unsqueeze(R_vis[:, 2, :, :], 1)
        R_VIS_R_grad_x, R_VIS_R_grad_y = Sobelxy(R_VIS_R)
        R_VIS_G_grad_x, R_VIS_G_grad_y = Sobelxy(R_VIS_G)
        R_VIS_B_grad_x, R_VIS_B_grad_y = Sobelxy(R_VIS_B)
        R_VIS_grad_x = torch.cat([R_VIS_R_grad_x, R_VIS_G_grad_x, R_VIS_B_grad_x], 1)
        R_VIS_grad_y = torch.cat([R_VIS_R_grad_y, R_VIS_G_grad_y, R_VIS_B_grad_y], 1)

        R_IR_R = torch.unsqueeze(R_ir[:, 0, :, :], 1)
        R_IR_G = torch.unsqueeze(R_ir[:, 1, :, :], 1)
        R_IR_B = torch.unsqueeze(R_ir[:, 2, :, :], 1)
        R_IR_R_grad_x, R_IR_R_grad_y = Sobelxy(R_IR_R)
        R_IR_G_grad_x, R_IR_G_grad_y = Sobelxy(R_IR_G)
        R_IR_B_grad_x, R_IR_B_grad_y = Sobelxy(R_IR_B)
        R_IR_grad_x = torch.cat([R_IR_R_grad_x, R_IR_G_grad_x, R_IR_B_grad_x], 1)
        R_IR_grad_y = torch.cat([R_IR_R_grad_y, R_IR_G_grad_y, R_IR_B_grad_y], 1)

        joint_grad_x = torch.maximum(R_VIS_grad_x, R_IR_grad_x)
        joint_grad_y = torch.maximum(R_VIS_grad_y, R_IR_grad_y)
        joint_int = torch.maximum(R_vis, R_ir)

        # TODO: max_int_loss or each_int_loss;if only Y channel
        max_int_loss = self.l1_loss(Fuse, joint_int)
        gradient_loss = 0.5 * self.l1_loss(Fuse_grad_x, joint_grad_x) + 0.5 * self.l1_loss(Fuse_grad_y, joint_grad_y)
        color_loss = self.l1_loss(Cb_Fuse, Cb_R_vis) + self.l1_loss(Cr_Fuse, Cr_R_vis)

        fusion_loss_total = 0.5 * max_int_loss + 0.2 * gradient_loss + 1 * color_loss

        # con_loss = self.l1_loss(Fuse, R_vis) + self.l1_loss(Fuse, R_ir)
        # con_loss = 0.5 * self.l1_loss(Y_Fuse, Y_R_vis) + 0.5 * self.l1_loss(Y_Fuse, R_ir[:, 0, :, :])
        # fusion_loss_total = 0.5 * max_int_loss + 0.5 * con_loss + 0.2 * gradient_loss + 1 * color_loss

        return fusion_loss_total


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.01, losses=['labels', 'boxes', 'cardinality']):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # 最后一个背景类,会较多乘以权重0.1避免影响
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        #     """Compute the losses related to the masks: the focal loss and the dice loss.
        #        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        #     """
        #     assert "pred_masks" in outputs
        #
        #     src_idx = self._get_src_permutation_idx(indices)
        #     tgt_idx = self._get_tgt_permutation_idx(indices)
        #     src_masks = outputs["pred_masks"]
        #     src_masks = src_masks[src_idx]
        #     masks = [t["masks"] for t in targets]
        #     # TODO use valid to mask invalid areas due to padding in loss
        #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        #     target_masks = target_masks.to(src_masks)
        #     target_masks = target_masks[tgt_idx]
        #
        #     # upsample predictions to the target size
        #     src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
        #                             mode="bilinear", align_corners=False)
        #     src_masks = src_masks[:, 0].flatten(1)
        #
        #     target_masks = target_masks.flatten(1)
        #     target_masks = target_masks.view(src_masks.shape)
        #     losses = {
        #         "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
        #         "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        #     }
        return 0

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      每张图片包含以下信息：'boxes'、'labels'、'image_id'、'area'、'iscrowd'、'orig_size'、'size'
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                     target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    # def __init__(self, model, tal_topk=10):  # model must be de-paralleled
    def __init__(self, nc=6, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        # device = next(model.parameters()).device  # get model device
        # h = model.args  # hyperparameters
        #
        # m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.hyp = h
        # self.stride = m.stride
        # self.nc = m.nc  # number of classes
        # self.no = m.nc + m.reg_max * 4
        # self.reg_max = m.reg_max
        # self.device = device

        self.hyp = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
        self.stride = torch.tensor([8, 16, 32])  # model strides=[8,16,32]这里相对输入图像尺寸的缩放比例,80*80,40*40,40*60相对640*640
        self.nc = nc  # number of classes
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.device = 'cuda:0'
        self.use_dfl = self.reg_max > 1

        from ultralytics.utils.tal import TaskAlignedAssigner
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        preds: list[tensor*3]
            [[bs,4*16+num_classes,80,80],
            [bs,4*16+num_classes,40,40],
            [bs,4*16+num_classes,20,20]]
        batch: dict{}
            'ori_shape': Tuple((768,1024),...*16)
            'resized_shape': Tuple((640,640),...*16)
            'img': Tensor[bs,3,h,w]
            'cls': Tensor[num_objects,1]
            'bboxes': Tensor[num_objects,4]
            'batch_idx': Tensor[num_objects]=tensor[0,0,0,0,0,1,1,2,2,2,2,2,....] 每个gt框对应的图像id索引
        """
        # in coco dataset
        # 1.预处理preds
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # bs 64,8400;bs 80,8400

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # bs 8400 80
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # bs 8400 64

        # 2.生成anchor_points
        dtype = pred_scores.dtype  # float32
        batch_size = pred_scores.shape[0]  # 16
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[
            0]  # image size (h,w)=(640,640)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # [8400,2],[8400,1],anchor坐标和缩放比例

        # 3.预处理targets gt
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
                            1)  # [num_objects,6]
        # 将targets转换为[bs max_id_num_objects 5(cls,bbox)];bs,某一张图最多的gt目标数,bbox从0-1归化到了原图640*640尺寸坐标下,并且xywh转为xyxy
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # [bs,max_id_num_objects,cls], [bs,max_id_num_objects,xyxy]
        # [bs max_id_num_objects 1=True/False],生成mask判断哪个bbox是有效的,因为之前都生成了max_id_num_objects,只有前面目标gt数的才有效
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes,解码dfl 4*16到4
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)  bs 8400 4

        # 4.正样本匹配,选取落在gt框Top10匹配分数的正样本,target_bboxes=[bs 8400 64],target_scores=[bs 8400 80],fg_mask正样本mask
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)  # 求target_scores和,用于损失平均;30多

        # Cls loss,类别损失3000多?
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss;ciou+dfl
        if fg_mask.sum():
            target_bboxes /= stride_tensor  # 缩放到特征尺度下
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp['dfl']  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class MakeLoss(nn.Module):
    def __init__(self, background):
        super(MakeLoss, self).__init__()

        self.semantic_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=background)
        # self.detect_loss = SetCriterion(num_classes=6, matcher=matcher.HungarianMatcher(1, 5, 2),
        #                                 weight_dict=self.get_detr_loss_dict())
        self.detect_loss = v8DetectionLoss(nc=6, tal_topk=10)
        self.FusionLoss = FusionLoss()

    def get_detr_loss_dict(self, detect_aux_loss=False):
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
        weight_dict['loss_giou'] = 2
        if detect_aux_loss:
            aux_weight_dict = {}
            dec_layers = 4
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        return weight_dict

    def forward(self, inputs, outputs, Mask, seg_label=None, detect_label=None, train_mode=1):
        input_vis, input_ir = inputs
        out_semantic, out_detection, Fus_img = outputs

        loss = torch.tensor(0)
        fusion_loss_total = torch.tensor(0)
        semantic_loss_total = torch.tensor(0)
        detection_loss_total = torch.tensor(0)

        if train_mode == 1:
            # 只训融合
            fusion_loss_total = self.FusionLoss(inputs, Fus_img, Mask)
            loss = fusion_loss_total
        elif train_mode == 2:
            # 只训检测
            # # 1.detr 检测损失
            # detection_loss_dict = self.detect_loss(out_detection, detect_label)
            # # 权重系数 {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
            # weight_dict = self.detect_loss.weight_dict
            # # 总损失 = 回归损失：loss_bbox（L1）+loss_bbox  +   分类损失：loss_ce
            # detection_loss_total = sum(
            #     detection_loss_dict[k] * weight_dict[k] for k in detection_loss_dict.keys() if k in weight_dict)
            # 2.yolo 检测损失
            detection_loss_total, detection_loss_detach = self.detect_loss(out_detection, detect_label)
            loss = 0.01 * detection_loss_total
        elif train_mode == 3:
            # 只训分割
            semantic_loss_total = self.semantic_loss(out_semantic, seg_label)
            loss = semantic_loss_total
        elif train_mode == 4:
            # 融合+检测
            fusion_loss_total = self.FusionLoss(inputs, Fus_img, Mask)
            detection_loss_total, detection_loss_detach = self.detect_loss(out_detection, detect_label)
            # TODO: 损失比重可能不好导致白边,0.1/1 * fusion_loss_total
            loss = 0.01 * detection_loss_total + 0.1 * fusion_loss_total
        elif train_mode == 5:
            # 融合+分割
            fusion_loss_total = self.FusionLoss(inputs, Fus_img, Mask)
            semantic_loss_total = self.semantic_loss(out_semantic, seg_label)
            loss = 1 * semantic_loss_total + 0.1 * fusion_loss_total
        elif train_mode == 6:
            # 融合+检测/分割
            fusion_loss_total = self.FusionLoss(inputs, Fus_img, Mask)
            if detect_label:
                detection_loss_total, detection_loss_detach = self.detect_loss(out_detection, detect_label)
            if seg_label:
                semantic_loss_total = self.semantic_loss(out_semantic, seg_label)
            loss = 0.01 * detection_loss_total + 0.1 * fusion_loss_total + 1 * semantic_loss_total

        return loss, semantic_loss_total, detection_loss_total, fusion_loss_total
