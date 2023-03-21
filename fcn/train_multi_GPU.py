import time
import os
import datetime

import torch

from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
import transforms as T
import numpy as np
import random

from Datasets.dataset_build import Pre_datasets
from Models.model_build import create_model

# 远程调试
# import debugpy; debugpy.connect(('10.59.139.1', 5678))

import wandb
from train_utils.distributed_utils import is_main_process


def main(args):
    # 分布式训练初始化
    init_distributed_mode(args)
    print(args.name_date)
    print(args)
    device = torch.device(args.device)

    # segmentation nun_classes(background)
    num_classes = args.num_classes

    # 用来保存运行结果的文件，只在主进程上进行写操作
    results_log = args.checkpoint_dir + "/output.log"
    results_csv = args.checkpoint_dir + "/metadata.csv"
    if args.rank in [-1, 0]:
        # write into csv
        with open(results_csv, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"epoch,mean_loss,mIOU,acc_global,lr\n" 
            f.write(train_info)

    # 对datasets进行预处理
    train_data_loader, val_data_loader, train_sampler = Pre_datasets(args)
    
    print("Creating model")
    # create model num_classes equal background + 20 classes
    model = create_model(args)
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.ddp)
        model_without_ddp = model.module

    # 设置参数的学习率
    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10}) 
    if args.contrast != -1:
        if args.loss_name == 'simsiam':
            params_simsiam = [p for p in model_without_ddp.contrast.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params_simsiam, "lr": args.lr * 10})
        elif args.loss_name == "intra":
            if args.L3_loss != 0:
                params_L3u = [p for p in model_without_ddp.ProjectorHead_3u.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params_L3u, "lr": args.lr * 10})
            if args.L2_loss != 0:
                params_L2u = [p for p in model_without_ddp.ProjectorHead_2u.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params_L2u, "lr": args.lr * 10})
            if args.L1_loss != 0:
                params_L1u = [p for p in model_without_ddp.ProjectorHead_1u.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params_L1u, "lr": args.lr * 10})
        else:
            if args.L3_loss != 0:
                params_L3d = [p for p in model_without_ddp.ProjectorHead_3d.parameters() if p.requires_grad]
                params_L3u = [p for p in model_without_ddp.ProjectorHead_3u.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params_L3d, "lr": args.lr * 10})
                params_to_optimize.append({"params": params_L3u, "lr": args.lr * 10})
            if args.L2_loss != 0:
                params_L2d = [p for p in model_without_ddp.ProjectorHead_2d.parameters() if p.requires_grad]
                params_L2u = [p for p in model_without_ddp.ProjectorHead_2u.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params_L2d, "lr": args.lr * 10})
                params_to_optimize.append({"params": params_L2u, "lr": args.lr * 10})
            if args.L1_loss != 0:
                params_L1d = [p for p in model_without_ddp.ProjectorHead_1d.parameters() if p.requires_grad]
                params_L1u = [p for p in model_without_ddp.ProjectorHead_1u.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params_L1d, "lr": args.lr * 10})
                params_to_optimize.append({"params": params_L1u, "lr": args.lr * 10})
            
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # 梯度加速
    K = args.GAcc
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader)//K, args.epochs, warmup=True)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    if args.test_only:
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        return

    ##### wandb #####
    if args.wandb and (args.rank in [-1, 0]):
        os.environ["WANDB_API_KEY"] = 'ae69f83abb637683132c012cd248d4a14177cd36'
        os.environ['WANDB_MODE'] = args.wandb_model
        wandb.init(project="DCNet")
        wandb.config.update(args)
        wandb.watch(model, log="all", log_freq=10) # 上传梯度信息

    print(model)
    best_IOU = 0
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = train_one_epoch(args, model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes, epoch=epoch)
        acc_global, acc, iu = confmat.compute()
        IOU = iu.mean().item() * 100
        val_info = (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                IOU)
        # val_info = str(confmat) # 修改展开了
        print(val_info)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            # write into txt
            with open(results_csv, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"{epoch},{mean_loss},{IOU},{acc_global},{lr}\n" 
                f.write(train_info)
          
        
        if args.checkpoint_dir:
            # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
            mkdir(args.checkpoint_dir)
            # 只在主节点上执行保存权重操作
            save_file = {'model': model_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'args': args,
                         'epoch': epoch}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            save_on_master(save_file,
                            '{}/checkpoints/model_latest.pth'.format(args.checkpoint_dir))
            if IOU > best_IOU:
                best_IOU = IOU
                save_on_master(save_file,
                            '{}/checkpoints/model_best.pth'.format(args.checkpoint_dir))

        ##### wandb #####
        if args.wandb and (args.rank in [-1, 0]):
            wandb.log({"mean_loss": mean_loss, "mIOU": IOU, "best_IOU": best_IOU, "acc_global": acc_global, "lr": lr, "epoch": epoch})
      

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 只在主节点上保存
    if is_main_process() and args.wandb:
            # batch_size = 1
            # input_shape = (3, 480, 480)
            # x = torch.randn(batch_size,*input_shape).cuda()
            # torch.onnx.export(model.module, x, "{}/model_{}.onnx".format(args.checkpoint_dir, epoch))
            # wandb.save("{}/model_{}.onnx".format(args.checkpoint_dir, epoch))

            # wandb.save('{}/checkpoints/model_{}.pth'.format(args.checkpoint_dir, epoch))

            wandb.save(results_csv)
            wandb.save(results_log)


def str2bool(v):
    """Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed) 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(VOCdevkit)
    parser.add_argument('--data_path', default='pascal-voc-2012', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=21, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('--batch_size', default=16, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--batch_size_val', default=8, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument("--aux", default=False, type=str2bool, help="auxilier loss")
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=str2bool, default=False, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.0001，如果效果不好可以尝试加大学习率
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--checkpoint_dir', default='./results', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=str2bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--seed", default=304, type=int,
                        help="random seed")

    parser.add_argument("--name_date", default='experiment_name/date', type=str,
                        help="save file for result")

    # wandb设置
    parser.add_argument('--wandb', default=False, type=str2bool, help='w/o wandb')
    parser.add_argument('--wandb_model', default='dryrun', type=str, help='run or dryrun')

    # DCNet专属设计
    parser.add_argument('--model_name', default='fcn_resnet50', type=str, help='fcn_resnet50 dcnet_resnet50')
    parser.add_argument("--project_dim", default=128, type=int, help="the dim of projector")
    parser.add_argument("--loss_name", default="intra", type=str, help="segloss intra inter double")
    parser.add_argument("--contrast", default=10, type=int, help="epoch start with contrast")
    parser.add_argument("--pre_trained", default="fcn_resnet50_coco.pth", type=str, help="pre_trained name")
    parser.add_argument("--L3_loss", default=0, type=float, help="L3 loss")
    parser.add_argument("--L2_loss", default=0, type=float, help="L2 loss")
    parser.add_argument("--L1_loss", default=0, type=float, help="L1 loss")
    parser.add_argument("--GAcc", default=1, type=int, help="Gradient Accumulation")
    parser.add_argument("--memory_size", default=5000, type=int, help="")
    parser.add_argument("--proj_dim", default=128, type=int, help="")
    parser.add_argument("--network_stride", default=8, type=int, help="")
    parser.add_argument("--pixel_update_freq", default=10, type=int, help="")
    parser.add_argument('--ddp', default=False, type=str2bool, help='')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.checkpoint_dir:
        args.checkpoint_dir = args.checkpoint_dir + "/" + args.name_date
        mkdir(args.checkpoint_dir)
        mkdir(args.checkpoint_dir + "/checkpoints")

    main(args)
