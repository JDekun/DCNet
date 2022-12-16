import torch
from torch import nn
import train_utils.distributed_utils as utils
import torch.nn.functional as F 
from train_utils.intra_contrastive_loss import  IntraPixelContrastLoss
from train_utils.inter_contrastive_loss import  InterPixelContrastLoss
from train_utils.double_contrastive_loss import  DoublePixelContrastLoss
from train_utils.double_contrastive_selfpace_loss import  SELFPACEDoublePixelContrastLoss
from train_utils.double_contrastive_selfpace_epoch_loss import  EPOCHSELFPACEDoublePixelContrastLoss
from contextlib import nullcontext
from collections import OrderedDict

def criterion(args, inputs, target, epoch):
    losses = {}
    loss_name = args.loss_name
    epochs = args.epochs
    
    if args.model_name == "fcn_resnet50" or args.contrast == -1:
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
            else:
                # proj_x = x[0]
                proj_y = x[1]
                

                h, w = proj_y.shape[2], proj_y.shape[3]
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
    
    loss = losses['out']

    for name, x in losses.items():
        if name != "out" and args.contrast > epoch:
            loss += 0 * x
        elif name != "out":
            loss += x
    
    return loss

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 10, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat

def dequeue_and_enqueue(args, keys, key_y, labels,
                        encode_queue, encode_queue_ptr,
                        decode_queue, decode_queue_ptr):
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]

    labels = labels[:, ::args.network_stride, ::args.network_stride]

    for bs in range(batch_size):
        this_feat = keys[bs].contiguous().view(feat_dim, -1)
        this_feat_y = key_y[bs].contiguous().view(feat_dim, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0 and x != 255]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero()

            # segment enqueue and dequeue
            # feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
            # ptr = int(encode_queue_ptr[lb])
            # encode_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
            # encode_queue_ptr[lb] = (encode_queue_ptr[lb] + 1) % args.memory_size

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, args.pixel_update_freq)
            # feat = feat[:, perm[:K]]
            feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)
            feat_y = this_feat_y[:, perm[:K]]
            feat_y = torch.transpose(feat_y, 0, 1)
            ptr = int(decode_queue_ptr[lb])

            if ptr + K >= args.memory_size:
                decode_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                decode_queue_ptr[lb] = 0
                encode_queue[lb, -K:, :] = nn.functional.normalize(feat_y, p=2, dim=1)
                encode_queue[lb] = 0
            else:
                decode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                decode_queue_ptr[lb] = (decode_queue_ptr[lb] + 1) % args.memory_size
                encode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat_y, p=2, dim=1)
                encode_queue_ptr[lb] = (encode_queue_ptr[lb] + 1) % args.memory_size

def train_one_epoch(args, model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    i = 1
    K = args.GAcc
    optimizer.zero_grad()
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        my_context = model.no_sync if args.rank != -1 and i % K != 0 else nullcontext
        with my_context():
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image, target)
                if args.contrast != -1 and args.memory_size >0:
                    if args.L3_loss != 0:
                        result3 = OrderedDict()
                        result3['encode_queue'] = model.module.encode3_queue
                        result3['encode_queue_ptr'] = model.module.encode3_queue_ptr
                        result3['decode_queue'] = model.module.decode3_queue
                        result3['decode_queue_ptr'] = model.module.decode3_queue_ptr
                        output["L3"].append(result3)
                    if args.L2_loss != 0:
                        result2 = OrderedDict()
                        result2['encode_queue'] = model.module.encode2_queue
                        result2['encode_queue_ptr'] = model.module.encode2_queue_ptr
                        result2['decode_queue'] = model.module.decode2_queue
                        result2['decode_queue_ptr'] = model.module.decode2_queue_ptr
                        output["L2"].append(result2)
                    if args.L1_loss != 0:
                        result1 = OrderedDict()
                        result1['encode_queue'] = model.module.encode1_queue
                        result1['encode_queue_ptr'] = model.module.encode1_queue_ptr
                        result1['decode_queue'] = model.module.decode1_queue
                        result1['decode_queue_ptr'] = model.module.decode1_queue_ptr
                        output["L1"].append(result1)
                loss = criterion(args, output, target, epoch)
            
            if args.memory_size:
                # 更新队列
                if args.L3_loss != 0:
                    queue = output["L3"][5]
                    feats_que =  output["L3"][2]
                    feats_y_que =  output["L3"][3]
                    labels_que =  output["L3"][4]
                    dequeue_and_enqueue(args, feats_que, feats_y_que, labels_que,
                                        encode_queue=queue["encode_queue"],
                                        encode_queue_ptr=queue["encode_queue_ptr"],
                                        decode_queue=queue["decode_queue"],
                                        decode_queue_ptr=queue["decode_queue_ptr"])


            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if i % K == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
        i += 1

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
