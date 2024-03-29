from collections import OrderedDict

from typing import Dict, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .resnet_backbone import resnet50, resnet101
from .mobilenet_backbone import mobilenet_v3_large

from  Models.Attention.CBAM import CBAMBlock
from  Models.Attention.PSA import PSA
from  Models.Attention.SelfAttention import ScaledDotProductAttention


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, args, backbone, classifier, aux_classifier=None):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

        self.attention_name = args.attention
        
        num_classes = 1
        self.contrast = True if args.contrast != -1 else False

        if args.contrast != -1 and args.memory_size > 0:
            if args.L3_loss != 0:
                self.register_buffer("encode3_queue", nn.functional.normalize(torch.randn(num_classes, args.memory_size, args.project_dim), p=2, dim=2))
                self.register_buffer("encode3_queue_ptr", torch.zeros(num_classes, dtype=torch.long))
                self.register_buffer("code3_queue_label", torch.randn(num_classes, args.memory_size))
            
            if args.L2_loss != 0:
                self.register_buffer("encode2_queue", nn.functional.normalize(torch.randn(num_classes, args.memory_size, args.project_dim), p=2, dim=2))
                self.register_buffer("encode2_queue_ptr", torch.zeros(num_classes, dtype=torch.long))
                self.register_buffer("code2_queue_label", torch.randn(num_classes, args.memory_size))
            
            if args.L1_loss != 0:
                self.register_buffer("encode1_queue", nn.functional.normalize(torch.randn(num_classes, args.memory_size, args.project_dim), p=2, dim=2))
                self.register_buffer("encode1_queue_ptr", torch.zeros(num_classes, dtype=torch.long))
                self.register_buffer("code1_queue_label", torch.randn(num_classes, args.memory_size)) 

    def forward(self, x: Tensor, target=None, is_eval = False) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()

        x = features["out"]
        x = self.classifier(x)
        # 使用双线性插值还原回原图尺度
        out = F.interpolate(x["out"], size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = out

        # 对比simsiam模块
        if  (self.contrast ) and (is_eval == False):
            temp = x["aspp"]

            aspp_one = F.normalize(temp[0], dim=1)
            aspp_two = F.normalize(temp[1], dim=1)
            aspp_three = F.normalize(temp[2], dim=1)

            result["L1"] = [aspp_one, aspp_two]
            result["L2"] = [aspp_two, aspp_three]
            result["L3"] = [aspp_three, aspp_one]

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 使用双线性插值还原回原图尺度
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], contrast, attention, out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True))
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        if contrast != -1:
            self.contrast = contrast
            self.mep = contrast_head(256, 128, attention)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        _aspp = []
        _sum = []
        count = 0
        for conv in self.convs:
            temp = conv(x)
            if count >0 and count < 4:
                _aspp.append(temp)
            else:
                _res.append(temp)
            count += 1
            _sum.append(temp)
        
        if self.contrast != -1:
            down, up = self.mep(_aspp)
            for i in up:
                _res.append(i)
            res = torch.cat(_res, dim=1)
            return self.project(res), down
        else:
            sum = torch.cat(_sum, dim=1)
            return self.project(sum)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, contrast, attention) -> None:
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36], contrast, attention),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = OrderedDict()
        count = 0
        for modul in self:
            if count == 0:
                x, mep = modul(x)
            else:
                x = modul(x)
            count =count + 1
        out['out'] = x
        out['aspp'] = mep
        return out


class ASPPDown(nn.Sequential):
    def __init__(self, in_channels: int, pre_dim: int) -> None:
        super(ASPPDown, self).__init__(
            nn.Conv2d(in_channels, pre_dim, 1, bias=False),
            nn.BatchNorm2d(pre_dim),
            nn.ReLU(inplace=True)
        )

class ASPPUp(nn.Sequential):
    def __init__(self, pre_dim: int, in_channels: int) -> None:
        super(ASPPUp, self).__init__(
            nn.Conv2d(pre_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ) 

class contrast_head(nn.Module):
    def __init__(self, in_channels: int, pre_dim: int, attention) -> None:
        super(contrast_head, self).__init__()

        down = []
        up = []
        for i in range(3):
            down.append(ASPPDown(in_channels, pre_dim))
            up.append(ASPPUp(pre_dim, in_channels))

        self.down = nn.ModuleList(down)
        self.up = nn.ModuleList(up)

        if attention == "cbam":
            self.attention = CBAMBlock(channel=128,reduction=8,kernel_size=7)
        elif "selfattention" in attention:
            head = int(attention.split("_")[1])
            self.attention = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=head)
        self.attention_name = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        _con = []
        count = 0
        for conv in self.down:
            temp = conv(x[count])

            if self.attention_name == "cbam":
                temp = self.attention(temp)
            elif "selfattention" in self.attention_name:
                temp = self.attention(temp, temp, temp)
            
            _res.append(temp)
            count += 1
        cou = 0
        for con in self.up:
            temp = con(_res[cou])
            _con.append(temp)
            cou += 1
        return _res, _con
    
    
def mep_resnet50(args, aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth'
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    return_layers['layer1'] = 'contrast_en'
    if aux:
        return_layers['layer3'] = 'aux'
    # 重构backbone
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    if pretrain_backbone:
        print("loading resnet101-backnone weight...")
        missing_keys, unexpected_keys = backbone.load_state_dict(torch.load("../../input/pre-trained/resnet50-imagenet.pth", map_location='cpu'), strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes, num_classes, args.contrast, args.attention)

    model = DeepLabV3(args, backbone, classifier, aux_classifier)

    return model

def mep_resnet101(args, aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])
            
    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    return_layers['layer1'] = 'contrast_en'
    if aux:
        return_layers['layer3'] = 'aux'
    # 重构backbone
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        print("loading resnet101-backnone weight...")
        missing_keys, unexpected_keys = backbone.load_state_dict(torch.load("../../input/pre-trained/resnet101-imagenet.pth", map_location='cpu'), strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes, num_classes, args.contrast, args.attention)

    model = DeepLabV3(args, backbone, classifier, aux_classifier)

    return model