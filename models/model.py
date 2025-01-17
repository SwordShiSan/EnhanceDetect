import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.logger import get_logger

logger = get_logger()


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)
        elif isinstance(m, nn.Embedding):
            group_decay.append(m.weight)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


class MRFS(nn.Module):
    def __init__(self, cfg=None, criterion=None, norm_layer=nn.BatchNorm2d):
        super(MRFS, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        if cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoder_agg import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoder_agg import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoder_agg import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoder_agg import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            from .encoder_agg import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
            self.channels = [32, 64, 160, 256]
        else:
            logger.info('Using backbone: Segformer-B4')
            from .encoder_agg import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        logger.info('Using MLP Decoder SegHead')
        logger.info('Using Yolov8 DetectHead')
        # 分割头
        from .Seg_head import DecoderHead
        self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.seg_num_classes, norm_layer=norm_layer,
                                       embed_dim=cfg.decoder_embed_dim)
        # 检测头;no neck;with neck
        from .Detect_head import DetrHead, YoloHead, YoloDetectHead
        # self.detect_head = DetrHead(num_classes=6, hidden_dim=256)
        # self.detect_head = YoloDetectHead(nc=cfg.detect_num_classes, ch=self.channels[1:])
        self.detect_head = YoloHead(nc=cfg.detect_num_classes, ch=self.channels[1:])
        # 融合头
        from .Fusion_head import CNNHead
        self.aux_head = CNNHead(in_channels=self.channels)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_backbone)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                        self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')
        logger.info('Initing weights ...')

    def forward(self, rgb, modal_x):
        ori_size = rgb.shape
        ori_inputs = [rgb, modal_x]
        x_vision, x_semantic, x_detection = self.backbone(rgb, modal_x)

        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=ori_size[2:], mode='bilinear', align_corners=False)

        out_detection = self.detect_head(x_detection)

        fus_img = self.aux_head.forward(x_vision, ori_inputs)

        outputs = [out_semantic, out_detection, fus_img]
        return outputs

    def forward_loss(self, rgb, modal_x, Mask=None, seg_label=None, detect_label=None, train_mode=1):
        inputs = [rgb, modal_x]
        outputs = self.forward(rgb, modal_x)
        # train: return loss
        loss = self.criterion(inputs, outputs, Mask, seg_label, detect_label, train_mode)
        return loss
