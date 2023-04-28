import torch

import train_utils.distributed_utils as utils

from contextlib import nullcontext
from collections import OrderedDict
from train_utils.loss_manage import criterion


def train_one_epoch(args, model, optimizer, data_loader, device, epoch, epochs, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [{}/{}]'.format(epoch, epochs)

    i = 1
    K = args.GAcc
    optimizer.zero_grad()
    for image, target in metric_logger.log_every(data_loader, print_freq, header, epoch, epochs):
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
                        # result3['decode_queue'] = model.module.decode3_queue
                        # result3['decode_queue_ptr'] = model.module.decode3_queue_ptr
                        result3['code_queue_label'] = model.module.code3_queue_label
                        output["L3"].append(result3)
                    if args.L2_loss != 0:
                        result2 = OrderedDict()
                        result2['encode_queue'] = model.module.encode2_queue
                        result2['encode_queue_ptr'] = model.module.encode2_queue_ptr
                        # result2['decode_queue'] = model.module.decode2_queue
                        # result2['decode_queue_ptr'] = model.module.decode2_queue_ptr
                        result2['code_queue_label'] = model.module.code2_queue_label
                        output["L2"].append(result2)
                    if args.L1_loss != 0:
                        result1 = OrderedDict()
                        result1['encode_queue'] = model.module.encode1_queue
                        result1['encode_queue_ptr'] = model.module.encode1_queue_ptr
                        # result1['decode_queue'] = model.module.decode1_queue
                        # result1['decode_queue_ptr'] = model.module.decode1_queue_ptr
                        result1['code_queue_label'] = model.module.code1_queue_label
                        output["L1"].append(result1)
                loss = criterion(args, output, target, epoch)
            
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


def evaluate(model, data_loader, device, num_classes, epoch, epochs):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [{}/{}]'.format(epoch, epochs)
    
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 40, header, epoch, epochs):
            image, target = image.to(device), target.to(device)
            
            output = model(image, is_eval=True)
           
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


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
