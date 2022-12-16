from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbone import resnet50, resnet101


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

class Queue(nn.Module):
    def __init__(self,args,name):
        super(Queue, self).__init__()
        self.m = 0.999
        result = OrderedDict()

        num_classes = args.num_classes + 1
        dim = args.proj_dim
        self.r = args.memory_size
  
        result['encode_queue'] = nn.functional.normalize(self.register_buffer(f"encode{name}_queue", torch.randn(num_classes, self.r, dim)), p=2, dim=2)
        result['encode_queue_ptr'] = self.register_buffer("encode_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

        result['decode_queue'] =  nn.functional.normalize(self.register_buffer(f"decode{name}_queue", torch.randn(num_classes, self.r, dim)), p=2, dim=2)
        result['decode_queue_ptr'] = self.register_buffer("decode_queue_ptr", torch.zeros(num_classes, dtype=torch.long)) 

        return result

class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

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

    def __init__(self, args, backbone, classifier, aux_classifier=None, ProjectorHead=None):
        super(FCN, self).__init__()
        self.loss_name = args.loss_name
        self.contrast = args.contrast
        self.L3_loss = args.L3_loss
        self.L2_loss = args.L2_loss
        self.L1_loss = args.L1_loss

        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

        # self.m = 0.999
        self.r = args.memory_size
        # num_classes = args.num_classes + 1
        # dim = args.proj_dim

        if self.contrast != -1:
            if self.L3_loss != 0:
                self.ProjectorHead_3d = ProjectorHead["3d"]
                self.ProjectorHead_3u = ProjectorHead["3u"]
                if self.r:
                    self.queue3 = Queue(args, name = "L3")
            if self.L2_loss != 0:
                self.ProjectorHead_2d = ProjectorHead["2d"]
                self.ProjectorHead_2u = ProjectorHead["2u"]
                if self.r:
                    self.queue2 = Queue(args, name = "L2")
            if self.L1_loss != 0:
                self.ProjectorHead_1d = ProjectorHead["1d"]
                self.ProjectorHead_1u = ProjectorHead["1u"]
                if self.r:
                    self.queue1 = Queue(args, name = "L1")

            # if self.r:
            #     queue = Queue()
                # self.register_buffer("encode_queue", torch.randn(num_classes, self.r, dim))
                # self.segment_queue = nn.functional.normalize(self.encode_queue, p=2, dim=2)
                # self.register_buffer("encode_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

                # self.register_buffer("decode_queue", torch.randn(num_classes, self.r, dim))
                # self.pixel_queue = nn.functional.normalize(self.decode_queue, p=2, dim=2)
                # self.register_buffer("decode_queue_ptr", torch.zeros(num_classes, dtype=torch.long))               

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        classifer = self.classifier(x)
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        out = F.interpolate(classifer["cls"], size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = out

        # if self.aux_classifier is not None:
        #     x = features["aux"]
        #     L3a = self.aux_classifier(x)
        #     if not self.contrast:
        #     # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        #         x = F.interpolate(L3a, size=input_shape, mode='bilinear', align_corners=False)
        #         result["aux"] = x

        # if self.ProjectorHead is not None:
        if self.contrast != -1:
            if self.L3_loss != 0:
                L3d = features["L3d"]
                L3d = self.ProjectorHead_3d(L3d)
                L3d = F.normalize(L3d, p=2, dim=1)
                L3u = classifer["L3u"]
                L3u = self.ProjectorHead_3u(L3u)
                L3u = F.normalize(L3u, p=2, dim=1)
                if self.r:
                    queue = self.queue3
                    result["L3"] = [L3d, L3u, queue]
                else:
                    result["L3"] = [L3d, L3u]
            if self.L2_loss != 0:
                L2u = classifer["L2u"]
                L2u = self.ProjectorHead_2u(L2u)
                L2u = F.normalize(L2u, p=2, dim=1)
                L2d = features["L2d"]
                L2d = self.ProjectorHead_2d(L2d)
                L2d = F.normalize(L2d, p=2, dim=1)
                if self.r:
                    queue = self.queue2
                    result["L2"] = [L2d, L2u, queue]
                else:
                    result["L2"] = [L2d, L2u]
            if self.L1_loss != 0:
                L1d = features["L1d"]
                L1d = self.ProjectorHead_1d(L1d)
                L1d = F.normalize(L1d, p=2, dim=1)
                L1u = classifer["L1u"]
                L1u = self.ProjectorHead_1u(L1u)
                L1u = F.normalize(L1u, p=2, dim=1)
                result["L1"] = [L1d, L1u]
                if self.r:
                    queue = self.queue1
                    result["L1"] = [L1d, L1u, queue]
                else:
                    result["L1"] = [L1d, L1u]
                           
        return result

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class FCNHead(nn.Module):
    def __init__(self, in_channels, channels):
        super(FCNHead, self).__init__()
        L3u_channels = in_channels // 2
        L2u_channels = in_channels // 4
        L1u_channels = in_channels // 8
        self.L3u = nn.Sequential(
           DoubleConv(in_channels, L3u_channels)
            )
        self.L2u =nn.Sequential(
            DoubleConv(L3u_channels, L2u_channels)
            )
        self.L1u =nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(L2u_channels, L1u_channels),
            )
        self.cls =nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(L1u_channels, channels, 1))
       
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = OrderedDict()
        L3u = self.L3u(x)
        L2u = self.L2u(L3u)
        L1u = self.L1u(L2u)
        cls_ = self.cls(L1u)

        result["L3u"] =  L3u
        result["L2u"] =  L2u
        result["L1u"] =  L1u
        result["cls"] =  cls_
        
        return result


class FCNHead_aux(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead_aux, self).__init__(*layers)

class ProjectorHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 2
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        ]

        super(ProjectorHead, self).__init__(*layers)


def dcnet_resnet50(args, aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    project_dim = args.project_dim

    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    return_layers['layer3'] = 'L3d'
    return_layers['layer2'] = 'L2d'
    return_layers['layer1'] = 'L1d'
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead_aux(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    prejector = OrderedDict()
    if args.contrast != -1:
        if args.L3_loss != 0:
            Projector_3d = ProjectorHead(out_inplanes//2, project_dim)
            Projector_3u = ProjectorHead(out_inplanes//2, project_dim)
            prejector["3d"] = Projector_3d
            prejector["3u"] = Projector_3u
        if args.L2_loss != 0:
            Projector_2d = ProjectorHead(out_inplanes//4, project_dim)
            Projector_2u = ProjectorHead(out_inplanes//4, project_dim)
            prejector["2d"] = Projector_2d
            prejector["2u"] = Projector_2u
        if args.L1_loss != 0:
            Projector_1d = ProjectorHead(out_inplanes//8, project_dim)
            Projector_1u = ProjectorHead(out_inplanes//8, project_dim)
            prejector["1d"] = Projector_1d
            prejector["1u"] = Projector_1u

    model = FCN(args, backbone, classifier, aux_classifier, prejector)

    return model


def dcnet_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model
