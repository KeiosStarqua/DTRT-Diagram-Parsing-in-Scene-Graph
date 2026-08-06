# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrained: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class DynamicTokenPruner(nn.Module):
    """Token scorer that masks low-importance spatial tokens."""
    def __init__(self, embed_dim: int):
        super().__init__()
        hidden_dim = max(embed_dim // 2, 1)
        bottleneck_dim = max(embed_dim // 4, 1)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1)
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, keep_ratio: float):
        b, h, w, c = x.shape
        tokens = x.reshape(b, h * w, c)
        valid_tokens = ~padding_mask.reshape(b, h * w)
        projected = self.in_proj(tokens)
        local_tokens = projected[:, :, :c // 2]
        global_token = projected[:, :, c // 2:].mean(dim=1, keepdim=True)
        projected = torch.cat([local_tokens, global_token.expand(b, h * w, c - c // 2)], dim=-1)
        logits = self.score(projected).squeeze(-1)
        logits = logits.masked_fill(~valid_tokens, -torch.finfo(logits.dtype).max)

        hard_keep = torch.zeros_like(logits, dtype=torch.bool)
        valid_counts = valid_tokens.sum(dim=1)
        for batch_idx in range(b):
            if valid_counts[batch_idx] == 0:
                continue
            keep_count = torch.ceil(valid_counts[batch_idx].float() * keep_ratio).long()
            keep_count = torch.clamp(keep_count, min=1, max=valid_counts[batch_idx])
            keep_idx = torch.topk(logits[batch_idx], k=int(keep_count.item()), dim=0).indices
            hard_keep[batch_idx, keep_idx] = True

        if self.training:
            keep_prob = torch.sigmoid(logits)
            gate = hard_keep.float() - keep_prob.detach() + keep_prob
            gate = gate * valid_tokens.float()
        else:
            gate = hard_keep.float()

        pruned_mask = padding_mask | ~hard_keep.reshape(b, h, w)
        x = tokens * gate.unsqueeze(-1)
        return x.reshape(b, h, w, c), pruned_mask, hard_keep.reshape(b, h, w)


class ReconstructionHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = max(channels // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, drop_mask: torch.Tensor):
        return self.net(torch.cat([x, drop_mask.float().unsqueeze(1)], dim=1))


class SwinBackbone(nn.Module):
    """Torchvision Swin backbone with spatial token pruning."""
    _CHANNELS = {
        'swin_t': 768,
        'swin_s': 768,
        'swin_b': 1024,
        'swin_v2_t': 768,
        'swin_v2_s': 768,
        'swin_v2_b': 1024,
    }
    _STAGE_CHANNELS = {
        'swin_t': {1: 96, 3: 192, 5: 384, 7: 768},
        'swin_s': {1: 96, 3: 192, 5: 384, 7: 768},
        'swin_b': {1: 128, 3: 256, 5: 512, 7: 1024},
        'swin_v2_t': {1: 96, 3: 192, 5: 384, 7: 768},
        'swin_v2_s': {1: 96, 3: 192, 5: 384, 7: 768},
        'swin_v2_b': {1: 128, 3: 256, 5: 512, 7: 1024},
    }
    _WEIGHTS = {
        'swin_t': 'Swin_T_Weights',
        'swin_s': 'Swin_S_Weights',
        'swin_b': 'Swin_B_Weights',
        'swin_v2_t': 'Swin_V2_T_Weights',
        'swin_v2_s': 'Swin_V2_S_Weights',
        'swin_v2_b': 'Swin_V2_B_Weights',
    }

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool,
                 pretrained: bool, prune_keep_ratios: List[float], prune_stages: List[int],
                 enable_token_reconstruction: bool = False, token_recon_all_tokens: bool = False):
        super().__init__()
        weights = self._build_weights(name, pretrained)
        swin = getattr(torchvision.models, name)(weights=weights)
        self.body = swin.features
        self.num_channels = self._CHANNELS[name]
        self.return_interm_layers = return_interm_layers
        self.prune_stages = prune_stages
        self.prune_keep_ratios = prune_keep_ratios
        self.enable_token_reconstruction = enable_token_reconstruction
        self.token_recon_all_tokens = token_recon_all_tokens
        self.pruners = nn.ModuleDict({
            str(stage): DynamicTokenPruner(self._STAGE_CHANNELS[name][stage])
            for stage in prune_stages
        })
        self.recon_heads = nn.ModuleDict({
            str(stage): ReconstructionHead(self._STAGE_CHANNELS[name][stage])
            for stage in prune_stages
        }) if enable_token_reconstruction else None

        if not train_backbone:
            for parameter in self.body.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def _build_weights(cls, name: str, pretrained: bool):
        if not pretrained:
            return None
        weights_enum_name = cls._WEIGHTS[name]
        return getattr(torchvision.models, weights_enum_name).DEFAULT

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size):
        return F.interpolate(mask[:, None].float(), size=size, mode='nearest').to(torch.bool)[:, 0]

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None

        out: Dict[str, NestedTensor] = {}
        recon_outputs = []
        prune_idx = 0
        current_mask = mask
        for stage_idx, layer in enumerate(self.body):
            x = layer(x)
            current_mask = self._resize_mask(current_mask, x.shape[1:3])
            input_mask = self._resize_mask(mask, x.shape[1:3])
            current_mask = current_mask | input_mask

            if stage_idx in self.prune_stages:
                keep_ratio = self.prune_keep_ratios[prune_idx]
                stage_mask = current_mask
                pre_prune = x.permute(0, 3, 1, 2).contiguous()
                x, current_mask, keep_mask = self.pruners[str(stage_idx)](x, current_mask, keep_ratio)
                if self.enable_token_reconstruction:
                    drop_mask = (~keep_mask) & (~stage_mask)
                    recon_mask = (~stage_mask) if self.token_recon_all_tokens else drop_mask
                    recon_pred = self.recon_heads[str(stage_idx)](x.permute(0, 3, 1, 2).contiguous(), recon_mask)
                    recon_outputs.append({
                        'stage': stage_idx,
                        'pred': recon_pred,
                        'target': pre_prune.detach(),
                        'mask': recon_mask.unsqueeze(1),
                    })
                prune_idx += 1

            if self.return_interm_layers and stage_idx in (1, 3, 5, 7):
                out[str(len(out))] = NestedTensor(x.permute(0, 3, 1, 2).contiguous(), current_mask)

        if not self.return_interm_layers:
            out["0"] = NestedTensor(x.permute(0, 3, 1, 2).contiguous(), current_mask)
        if self.enable_token_reconstruction:
            return out, recon_outputs
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        extra_outputs = None
        if isinstance(xs, tuple):
            xs, extra_outputs = xs
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        if extra_outputs is not None:
            return out, pos, extra_outputs
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.return_interm_layers
    pretrained = is_main_process() and not args.no_backbone_pretrained
    if args.backbone.startswith('swin'):
        prune_keep_ratios = _parse_float_list(args.swin_prune_keep_ratios)
        prune_stages = _parse_int_list(args.swin_prune_stages)
        if args.disable_swin_token_pruning:
            prune_keep_ratios = []
            prune_stages = []
        if len(prune_keep_ratios) != len(prune_stages):
            raise ValueError("--swin_prune_keep_ratios and --swin_prune_stages must have the same length")
        invalid_ratios = [ratio for ratio in prune_keep_ratios if ratio <= 0 or ratio > 1]
        if invalid_ratios:
            raise ValueError("--swin_prune_keep_ratios values must be in the interval (0, 1]")
        if hasattr(torchvision.models, args.backbone):
            backbone = SwinBackbone(args.backbone, train_backbone, return_interm_layers,
                                    pretrained, prune_keep_ratios, prune_stages,
                                    enable_token_reconstruction=args.enable_token_reconstruction,
                                    token_recon_all_tokens=args.token_recon_all_tokens)
        else:
            raise ValueError(f"Unsupported Swin backbone: {args.backbone}")
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, pretrained)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def _parse_float_list(value: str):
    if value == "":
        return []
    return [float(item) for item in value.split(',')]


def _parse_int_list(value: str):
    if value == "":
        return []
    return [int(item) for item in value.split(',')]
