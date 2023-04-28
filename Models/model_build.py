import torch
import Models

def create_model(args):
    num_classes = args.num_classes
    aux = aux=args.aux
    model_name = args.model_name
    pre_trained = args.pre_trained
    

    if  pre_trained in ["resnet50-imagenet.pth", "resnet101-imagenet.pth"]:
        if  ("mep_res_" in model_name) or ("mep_sk_" in model_name):
            model = eval("Models."+model_name.rsplit("_",1)[0])(args, aux=aux, num_classes=num_classes, pretrain_backbone=True)
        else:
            model = eval("Models."+model_name)(args, aux=aux, num_classes=num_classes, pretrain_backbone=True)
    else:
        model = eval("Models."+model_name)(args, aux=aux, num_classes=num_classes, pretrain_backbone=False)

        weights_dict = torch.load(f"../../input/pre-trained/{pre_trained}", map_location='cpu')
            
        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]
                    
        if args.weight_only_backbone == True:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model