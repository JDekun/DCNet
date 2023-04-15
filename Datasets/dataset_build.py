from .pascal_voc import VOCSegmentation, get_transform
from .cityscapes_gf import Cityscapes
import os, torch

def Pre_datasets(args):
    if "../../input/" in args.data_path:
        data_path = args.data_path
    else:
        data_path = "../../input/" + args.data_path
    # check voc root
    if os.path.exists(os.path.join(data_path)) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(data_path))

    # load train data set
    train_dataset, val_dataset = datasets_load(args, data_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if 'pascal-voc-2012' in data_path :
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers,
            pin_memory=True, drop_last=True,
            collate_fn=train_dataset.collate_fn)

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size_val,
            sampler=test_sampler, num_workers=args.workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn)
    elif 'cityscapes' in data_path :
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers,
            pin_memory=True, drop_last=True)

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size_val,
            sampler=test_sampler, num_workers=args.workers,
            pin_memory=True)
    
    return train_data_loader, val_data_loader, train_sampler



def datasets_load(args, data_path):
    # load train data set
    if "pascal-voc-2012" in data_path:
        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
        train_dataset = VOCSegmentation(data_path,
                                        year="2012",
                                        transforms=get_transform(train=True),
                                        txt_name=args.data_train_type)
        # load validation data set
        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
        val_dataset = VOCSegmentation(data_path,
                                    year="2012",
                                    transforms=get_transform(train=False),
                                    txt_name="val.txt")
    elif "cityscapes" in data_path:
        crop_size = (1024, 512)
        # crop_size = (769, 769)
        # crop_size = (513, 513)
        train_dataset = Cityscapes(
                            root=data_path,
                            list_path="Datasets/list/"+args.data_train_type,
                            num_samples=None,
                            num_classes=19,
                            multi_scale=True,
                            flip=True,
                            ignore_label=255,
                            base_size=2048,
                            crop_size=crop_size,
                            downsample_rate=1,
                            scale_factor=16)
        
        test_size = (2048, 1024)
        # test_size = (769, 769)
        # test_size = (513, 513)
        val_dataset = Cityscapes(
                            root=data_path,
                            list_path="Datasets/list/val.txt",
                            num_samples=None,
                            num_classes=19,
                            multi_scale=False,
                            flip=False,
                            ignore_label=255,
                            base_size=2048,
                            crop_size=test_size,
                            downsample_rate=1)
    else:
        print("enrror datasets name")

    return train_dataset, val_dataset

