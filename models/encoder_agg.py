import copy
import math
import time
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import clip

from .modules import PCASC
from .modules import IGMAVC
from engine.logger import get_logger
from .position_encoding import PositionEmbeddingSine

logger = get_logger()


class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 多头自注意力
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head-> B C//num_head N num_heads
        # q=B num_head N C//num_head
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # k,v=B num_head N C//num_head
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # B num_head N C//num_head @ B num_head C//num_head N = B num_head N N
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # B num_head N N

        # B num_head N N @ B num_head N C//num_head -> B num_head N C//num_head -> B N num_head C//num_head -> B N C
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    input: B N C
    output: B N C 一维表示
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    将二维图像转为patch embedding一维表示,B C H W -> B N C
    TODO: 这里只进行了卷积下采样?改用PixShuffle?修改下采样倍数?
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # (224,224)
        patch_size = to_2tuple(patch_size)  # (7,7)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]  # H,W=32
        self.num_patches = self.H * self.W  # 1024个patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B C H W
        x = self.proj(x)  # B embed_dim H/4 W/4
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C(768)
        x = self.norm(x)

        return x, H, W


class DetrDetectDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # detect transformer decoder
        d_model = 256
        num_queries = 100
        hidden_dim = d_model
        nhead = 8
        dim_feedforward = 2048
        num_decoder_layers = 4

        self.position_emded = PositionEmbeddingSine(hidden_dim // 2)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # TODO 通道映射,将4个多尺度特征拼接一起;或者分别有4个decoder,4个输出 object query output最后再拼接映射成1个
        # B C=64 H/4 W/4 ;B C=128 H/8 W/8 ;B C=256 H/16 W/16 ;B C=512 H/32 W/32  -> B C=512 H/32 W/32
        # nn.PixelUnshuffle(r)下采样 C H*r W*r -> C*r*r H W
        self.input_proj = nn.ModuleList([
            nn.Conv2d(64, 8, kernel_size=1),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=1),
        ])
        self.input_proj_down = nn.ModuleList([
            nn.PixelUnshuffle(8),
            nn.PixelUnshuffle(4),
            nn.PixelUnshuffle(2),
            nn.PixelUnshuffle(1),
        ])
        # 这个当做类似encoder的输出,做位置编码?
        self.input_proj_out = nn.Conv2d(512 * 4, 256, kernel_size=1)  # B 512*4 H/32 W/32 -> B 256 H/32 W/32

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        decoder_norm = nn.LayerNorm(d_model)
        # 指定返回中间每个解码器的输出True,可用于损失计算aux_loss
        self.detect_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                                 return_intermediate=True)

    def forward(self, outs_semantic):
        # detr detect decoder output
        outs_detection = []
        for i, out_semantic in enumerate(outs_semantic):
            outs_detection.append(self.input_proj_down[i](self.input_proj[i](out_semantic)))
        outs_detection = torch.cat(outs_detection, dim=1)
        outs_detection = self.input_proj_out(outs_detection)  # 类比编码器输出,B 256 H/32 W/32
        pos = self.position_emded(outs_detection)  # 位置编码

        bs, c, h, w = outs_detection.shape
        outs_detection = outs_detection.flatten(2).permute(2, 0, 1)  # H*W/1024 B 256
        pos_embed = pos.flatten(2).permute(2, 0, 1)  # H*W/1024 B 256
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # 100 B 256
        tgt = torch.zeros_like(query_embed)  # 100 B 256
        # return_intermediate=True的话[4,b,100,256],否则[b,100,256]
        hs = self.detect_decoder(tgt, outs_detection, pos=pos_embed, query_pos=query_embed)  # layers=4 100 B 256
        hs = hs.transpose(1, 2)  # [4,b,100,256]
        return hs


class TransformerDecoder(nn.Module):
    '''
    需要：
    position_embedding:
    query_objects:
    tgt: torch.zeros_like(query_embed)初始化
    memory: encoder输入,语义特征输入,要patch_embedding
    mask:不传入
    '''

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


## Feature Modulation
'''
图像特征x = (1 + gamma) * x + beta,gamma和beta都是b c 1 1，它们在通道处发挥了作用
看法: 融合网络提取的每个通道都有不同的作用(抽象),而文本特征(编码'红外与可见光图像融合任务')指导哪一个通道对融合有帮助
进一步: 思考融合网络提取的每个通道对目标的特征都是不同的(比如人，车，房子，树等，每个通道侧重点不同),
可以考虑文本特征(编码'这是红外与可见光图像融合任务,希望使得人亮度更强，树纹理清晰'),来动态指导哪一个通道应该具有更高的权重
'''
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class RGBXTransformer(nn.Module):
    '''
    Struct: [ Encoder -> IGM-Att(to fusion) -> PC-Att(to seg) -> transformer decoder(to detect) ] x 4
    '''

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # pretrained clip
        # self.model_clip, _ = clip.load("ViT-B/32", device="cuda:0")
        # self.model_clip.eval()

        # patch_embed,extra_patch_embed分别可见光和红外图像
        # 初始输入B C H W
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])  # B C H W -> B H*W/16 C=64
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])  # B C H/4 W/4 -> B H*W/64 C=128
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])  # B C H/8 W/8 -> B H*W/256 C=256
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])  # B C H/16 W/16 -> B H*W/1024 C=512

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                    in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                    in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])

        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        cur += depths[3]

        self.IGMAVCs = nn.ModuleList([
            IGMAVC(dim=embed_dims[0], reduction=4),
            IGMAVC(dim=embed_dims[1], reduction=4),
            IGMAVC(dim=embed_dims[2], reduction=4),
            IGMAVC(dim=embed_dims[3], reduction=4)])

        self.PCASCs = nn.ModuleList([
            PCASC(dim=embed_dims[0], num_heads=num_heads[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, sr_ratio=sr_ratios[0]),
            PCASC(dim=embed_dims[1], num_heads=num_heads[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, sr_ratio=sr_ratios[1]),
            PCASC(dim=embed_dims[2], num_heads=num_heads[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, sr_ratio=sr_ratios[2]),
            PCASC(dim=embed_dims[3], num_heads=num_heads[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, sr_ratio=sr_ratios[3])])

        # detect transformer decoder
        # self.detr_detect_decoder = DetrDetectDecoder()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

    def forward_features(self, x_rgb, x_e, text=None):
        """
        x_rgb: B x N x H x W
        text: description
        """
        B = x_rgb.shape[0]
        outs_semantic = []
        outs_detection = []  # get stage 2 3 4,remove 1
        outs_vision = []

        # text feature;B 512
        # text_features = self.get_text_feature(text.expand(B, -1)).to(x_rgb.dtype)

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        # B H*W/16 C
        x_e, _, _ = self.extra_patch_embed1(x_e)
        for i, blk in enumerate(self.block1):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block1):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm1(x_rgb)
        x_e = self.extra_norm1(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B N C -> B C H W 转换后再通过这2个模块
        x_rgb, x_e = self.IGMAVCs[0](x_rgb, x_e)
        x_fused = self.PCASCs[0](x_rgb, x_e)  # b 64 120 160

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e, _, _ = self.extra_patch_embed2(x_e)
        for i, blk in enumerate(self.block2):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block2):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm2(x_rgb)
        x_e = self.extra_norm2(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.IGMAVCs[1](x_rgb, x_e)
        x_fused = self.PCASCs[1](x_rgb, x_e)  # b 128 60 80

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)
        outs_detection.append(x_fused)

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e, _, _ = self.extra_patch_embed3(x_e)
        for i, blk in enumerate(self.block3):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block3):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm3(x_rgb)
        x_e = self.extra_norm3(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.IGMAVCs[2](x_rgb, x_e)
        x_fused = self.PCASCs[2](x_rgb, x_e)  # b 256 30 40

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)
        outs_detection.append(x_fused)

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        for i, blk in enumerate(self.block4):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block4):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.IGMAVCs[3](x_rgb, x_e)
        x_fused = self.PCASCs[3](x_rgb, x_e)  # b 512 15 20

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)
        outs_detection.append(x_fused)

        # 1.detr detect decoder output
        # outs_detection = self.detr_detect_decoder(outs_semantic)
        # 2.yolo no decoder,direct out

        return outs_vision, outs_semantic, outs_detection

    def forward(self, x_rgb, x_e):
        out_vision, out_semantic, outs_detection = self.forward_features(x_rgb, x_e)
        return out_vision, out_semantic, outs_detection


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def load_dualpath_model(model, model_file):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            state_dict[k.replace('block', 'extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v

    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    del state_dict

    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))


class mit_b0(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4_ori(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b4_ori, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


# TODO
class mit_b4(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
