import torch
import torch.nn as nn
import timm
from timm import create_model
import torch.nn.functional as F
from typing import Optional, List
from backbones_unet import __available_models__

import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel

__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass



class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module




class Deeplabv3(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            encoder_freeze=False,
            pretrained=True,
            preprocessing=False,
            non_trainable_layers=(0, 1, 2, 3, 4),
            backbone_kwargs=None,
            backbone_indices=None,
            decoder_use_batchnorm=True,
            small_decoder=True,
            decoder_channels=(256, 128, 64, 32, 16),
            # decoder_channels=(512, 256, 128, 64, 32),
            in_channels=1,
            num_classes=5,
            # center=False,
            center=True,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU
    ):
        super().__init__()
        decoder_channels = (256, 128, 64, 32, 16) if small_decoder else (512, 256, 128, 64, 32)
        backbone_kwargs = backbone_kwargs or {}
        # NOTE some models need different backbone indices
        # specified based on the alignment of features
        # and some models won't have a full enough range
        # of feature strides to work properly.
        if backbone not in __available_models__ and backbone not in timm.list_models():
            raise RuntimeError(
                f"Unknown backbone {backbone}!\n"
                f"Existing models should be selected "
                f"from pretrained_backbones_unet import __available_models__"
            )

        # import ipdb;ipdb.set_trace()
        encoder = create_model(
            backbone, features_only=True, out_indices=backbone_indices,
            in_chans=in_channels, pretrained=pretrained, **backbone_kwargs
        )
        encoder_channels = [info["num_chs"] for info in encoder.feature_info][::-1]
        self.encoder = encoder
        if encoder_freeze: self._freeze_encoder(non_trainable_layers)
        if preprocessing:
            self.mean = self.encoder.default_cfg["mean"]
            self.std = self.encoder.default_cfg["std"]
        else:
            self.mean = None
            self.std = None

        if not decoder_use_batchnorm:
            norm_layer = None
        self.decoder = Deeplabv3Decoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
            activation=activation
        )

    def forward(self, x: torch.Tensor):
        # if self.mean and self.std:
        #    x = self._preprocess_input(x)
        x = self.encoder(x)
        # import ipdb;ipdb.set_trace()
        x.reverse()
        x = self.decoder(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        """
        Inference method. Switch model to `eval` mode,
        call `.forward(x)` with `torch.no_grad()`
        Parameters
        ----------
        x: torch.Tensor
            4D torch tensor with shape (batch_size, channels, height, width)
        Returns
        -------
        prediction: torch.Tensor
            4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training: self.eval()
        x = self.forward(x)
        return x

    def _freeze_encoder(self, non_trainable_layer_idxs):
        """
        Set selected layers non trainable, excluding BatchNormalization layers.
        Parameters
        ----------
        non_trainable_layer_idxs: tuple
            Specifies which layers are non-trainable for
            list of the non trainable layer names.
        """
        non_trainable_layers = [
            self.encoder.feature_info[layer_idx]["module"].replace(".", "_") \
            for layer_idx in non_trainable_layer_idxs
        ]
        for layer in non_trainable_layers:
            for child in self.encoder[layer].children():
                for param in child.parameters():
                    param.requires_grad = False
        return

    def _preprocess_input(self, x, input_range=[0, 1], inplace=False):
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {x.dtype}.")

        if x.ndim < 3:
            raise ValueError(
                f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {x.size()}"
            )

        if not inplace:
            x = x.clone()

        dtype = x.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=x.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        return x.sub_(mean).div_(std)


# UNet Blocks

class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = norm_layer(out_channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, trans_channels=None, scale_factor=2.0,
            activation=nn.ReLU, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, activation=activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = 0
        self.output_padding = 0 if scale_factor == 2 else 1
        if scale_factor != 2:
            self.padding = (scale_factor - 1) // 2
        self.trans_channels = trans_channels
        self.conv_t = nn.ConvTranspose2d(self.trans_channels, self.trans_channels, kernel_size=int(scale_factor),
                                         stride=2, padding=self.padding,
                                         output_padding=self.output_padding,
                                         groups=self.trans_channels) if scale_factor != 1 else nn.Identity()
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            x = self.conv_t(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x





class Deeplabv3Decoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=True,
            activation=nn.ReLU
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(
                channels, channels, scale_factor=1.0,
                activation=activation, norm_layer=norm_layer
            )
        else:
            self.center = nn.Identity()

        # list(decoder_channels[:-1][:len(encoder_channels)])
        in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
            [encoder_channels[0]] + list(decoder_channels[:-1]),
            list(encoder_channels[1:]) + [0])]

        # in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip([encoder_channels[0]] + list(decoder_channels), list(encoder_channels) + [0])]

        # import ipdb;ipdb.set_trace()
        # in_channels = [in_chs + skip_chs for in_chs, skip_chs in zip(
        #    [encoder_channels[0]] + list(decoder_channels),
        #   list(encoder_channels) + [0])]

        out_channels = decoder_channels

        transposed_channels = [encoder_channels[0], *list(decoder_channels)[:-1]]

        # import ipdb;ipdb.set_trace()

        if len(in_channels) != len(out_channels):
            in_channels.append(in_channels[-1] // 2)

        self.blocks = nn.ModuleList()
        for in_chs, out_chs, trans_chs in zip(in_channels, out_channels, transposed_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs, trans_channels=trans_chs, norm_layer=norm_layer))
        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        # skips = x
        # import ipdb;ipdb.set_trace()
        x = self.center(encoder_head)
        # x = self.center(F.max_pool2d(encoder_head, kernel_size=2))
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x