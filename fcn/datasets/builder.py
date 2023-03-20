from .pascal_voc import VOCSegmentation
from .cityscapes_gf import Cityscapes

import transforms as T


def datasets_load(args):
    # load train data set
    if "pascal-voc-2012" in args.data_path:
        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
        train_dataset = VOCSegmentation(args.data_path,
                                        year="2012",
                                        transforms=get_transform(train=True),
                                        txt_name="train.txt")
        # load validation data set
        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
        val_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=False),
                                    txt_name="val.txt")
    elif "cityscapes" in args.data_path:
        crop_size = (1024, 512)
        train_dataset = Cityscapes(
                            root=args.data_path,
                            list_path="datasets/list/train.lst",
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
        val_dataset = Cityscapes(
                            root=args.data_path,
                            list_path="datasets/list/val.lst",
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


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(train):
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)