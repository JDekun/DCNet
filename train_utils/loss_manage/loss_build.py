from torch import nn
import torch
import torch.nn.functional as F 
from .intra_contrastive_loss import  IntraPixelContrastLoss
from .inter_contrastive_loss import  InterPixelContrastLoss
from .double_contrastive_loss import  DoublePixelContrastLoss
from .double_contrastive_selfpace_loss import  SELFPACEDoublePixelContrastLoss
from .double_contrastive_selfpace_epoch_loss import  EPOCHSELFPACEDoublePixelContrastLoss
from .aspp_loss import  ASPP_CONTRAST_Loss
from .simsiam_loss import  simsiam_loss

def criterion(args, inputs, target, epoch):
    losses = {}
    loss_name = args.loss_name
    epochs = args.epochs
    
    if args.contrast == -1:
        for name, x in inputs.items():
            # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
        return losses['out']
    else:
        for name, x in inputs.items():
            # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
            if name == "out":
                pred_y = x
                losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
            elif name == "simsiam_loss":
                contrast_en = x["contrast_en"]
                contrast_de = x["contrast_de"]

                h, w = contrast_de.shape[2], contrast_de.shape[3]
                targ = target.unsqueeze(1).float().clone()
                targ = F.interpolate(targ, size=(h, w), mode='nearest')
                targ = targ.squeeze(1).long()

                criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
                loss = simsiam_loss(criterion, contrast_en, contrast_de, targ, ignore_index=255)
                losses[name] = loss
            else:
                proj_x = x[0]

                h, w = proj_x.shape[2], proj_x.shape[3]
                pred = F.interpolate(input=pred_y, size=(h, w), mode='bilinear', align_corners=False)
                _, predict = torch.max(pred, 1)
                
                # # 每层的语义分割像素交叉熵损失
                # h, w = target.size(1), target.size(2)
                # pred = F.interpolate(input=pred_y, size=(h, w), mode='bilinear', align_corners=False)
                # loss = nn.functional.cross_entropy(pred, target, ignore_index=255)

                # 层内对比损失
                if loss_name == "intra":
                    loss_contrast = IntraPixelContrastLoss(x, target, predict)
                elif loss_name == "inter":
                    loss_contrast = InterPixelContrastLoss(x, target, predict)
                elif loss_name == "double":
                    # loss_contrast = DoublePixelContrastLoss(x, target, predict)
                    # loss_contrast = SELFPACEDoublePixelContrastLoss(x, target, predict)
                    loss_contrast = EPOCHSELFPACEDoublePixelContrastLoss(args, epoch, epochs, x, target, predict)
                elif loss_name == "aspp_loss":
                    loss_contrast = ASPP_CONTRAST_Loss(args, epoch, epochs, x, target, predict)
                else:
                    print("the name of loss is None !!!")

                if name == "L1":
                    contrast_loss = args.L1_loss
                elif name == "L2":
                    contrast_loss = args.L2_loss
                elif name == "L3":
                    contrast_loss = args.L3_loss
                losses[name] = loss_contrast * contrast_loss

    if len(losses) == 1:
        return losses['out']
    
    if len(losses) == 2:
        return losses['out'] + losses['simsiam_loss']
    
    loss = losses['out']

    for name, x in losses.items():
        if name != "out" and args.contrast > epoch:
            loss += 0 * x
        elif name != "out":
            loss += x
    
    return loss