import time
import os
import datetime

import torch

from src import UNet, DC_UNet, DCNet, VGG16UNet, MobileV3Unet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import DriveDataset
import transforms as T
import numpy as np
import random

import wandb
from train_utils.distributed_utils import is_main_process

# 远程调试
# import debugpy; debugpy.connect(('10.59.139.1', 5678))

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes,model_name):
    if model_name == "DCNet":
        model = DCNet(in_channels=3, num_classes=num_classes, base_c=64, proj_d = 128)
    elif model_name == "DC_UNet":
        model = DC_UNet(in_channels=3, num_classes=num_classes, base_c=64, proj_d = 128)
    elif model_name == "UNet":
        model = UNet(in_channels=3, num_classes=num_classes, base_c=64)
    elif model_name == "VGG16UNet":
        model = VGG16UNet(num_classes=num_classes)
    elif model_name == "MobileV3Unet":
        model = MobileV3Unet(num_classes=num_classes)
    else:
        print("without this model")
    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    batch_size = args.batch_size
    layer_loss = args.layer_loss
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # special for DCNet
    with_contrast = args.with_contrast
    model_name = args.model_name
    loss_name = args.loss_name
    

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # # 用来保存coco_info的文件
    # results_file = "./output/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_root = args.data_path
    # check data root
    if os.path.exists(os.path.join(data_root, "DRIVE")) is False:
        raise FileNotFoundError("DRIVE dose not in path:'{}'.".format(data_root))

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size_val,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + foreground classes
    model = create_model(num_classes=num_classes, model_name=model_name)
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        return


    ##### wandb #####
    if args.wandb and (args.rank in [-1, 0]):
        os.environ["WANDB_API_KEY"] = 'ae69f83abb637683132c012cd248d4a14177cd36'
        os.environ['WANDB_MODE'] = args.wandb_model
        wandb.init(project="unet")
        wandb.config.update(args)
        wandb.watch(model, log="all", log_freq=10)

    best_dice = 0.
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler,batch_size=batch_size, 
                                        with_contrast = with_contrast, layer_loss=layer_loss, loss_name=loss_name)

        confmat, dice = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        acc_global, acc, iu = confmat.compute()
        val_info = (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)
        # val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")

        ##### wandb #####
        if args.wandb and (args.rank in [-1, 0]):
            wandb.log({"mean_loss": mean_loss, "mIOU": iu.mean().item() * 100, "acc_global": acc_global, "lr": lr, "dice": dice})

        # # 只在主进程上进行写操作
        # if args.rank in [-1, 0]:
        #     # write into txt
        #     with open(results_file, "a") as f:
        #         # 记录每个epoch对应的train_loss、lr以及验证集各指标
        #         train_info = f"[epoch: {epoch}]\n" \
        #                      f"train_loss: {mean_loss:.4f}\n" \
        #                      f"lr: {lr:.6f}\n" \
        #                      f"dice coefficient: {dice:.3f}\n"
        #         f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        if args.output_dir:
            # 只在主节点上执行保存权重操作
            save_file = {'model': model_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'args': args,
                         'epoch': epoch}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()

            if args.save_best is True:
                save_on_master(save_file,
                               os.path.join(args.output_dir, 'best_model_{}_{}.pth'.format(model_name, loss_name)))
            else:
                save_on_master(save_file,
                               os.path.join(args.output_dir, 'model_{}_{}_{}.pth'.format(epoch, model_name, loss_name)))
        
    # 只在主节点上保存
    if is_main_process():
            batch_size = 1
            input_shape = (3, 480, 480)
            x = torch.randn(batch_size,*input_shape).cuda()
            torch.onnx.export(model.module, x, "./{}/best_model_{}_{}.onnx".format(args.output_dir, model_name, loss_name))
            wandb.save("./{}/best_model_{}_{}.onnx".format(args.output_dir, model_name, loss_name))

            if args.save_best is True:
                wandb.save(os.path.join(args.output_dir, 'best_model_{}_{}.pth'.format(model_name, loss_name)))
            else:
                wandb.save(os.path.join(args.output_dir, 'model_{}_{}_{}.pth'.format(epoch, model_name, loss_name)))

            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

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

    # 训练文件的根目录(DRIVE)
    parser.add_argument('--data-path', default='../../../input/drive', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('--batch_size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--batch_size_val', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=str2bool, default=False, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.01(使用n块GPU建议乘以n)，如果效果不好可以尝试修改学习率
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 只保存dice coefficient值最高的权重
    parser.add_argument('--save-best', default=True, type=str2bool, help='only save best weights')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
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
    
    # 双对比损失设置
    parser.add_argument("--layer_loss", nargs=4, default=[0.1, 0.1, 0.05, 0.01], type=float,
                        help="loss for layers")
    parser.add_argument("--with_contrast", default=20, type=int,
                        help="when start epoch")
    parser.add_argument("--model_name", default="DC_UNet", type=str,
                        help="UNet DC_UNet DCNet VGG16UNet MobileV3Unet")
    parser.add_argument("--loss_name", default="intra", type=str,
                        help="segloss intra inter double")
    parser.add_argument("--seed", default=200, type=int,
                        help="random seed")
    
    # wandb设置
    parser.add_argument('--wandb', default=False, type=str2bool, help='w/o wandb')
    parser.add_argument('--wandb_model', default='dryrun', type=str, help='run or dryrun')

    args = parser.parse_args()

    set_seed(args.seed)

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
