import torch


def optim_manage(args, model_without_ddp):

    params_to_optimize = []
    # backbone
    params = [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]
    params_to_optimize.append({"params": params, "lr": args.lr})

    # decode
    params = [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]
    params_to_optimize.append({"params": params, "lr": args.lr })
    # if model_without_ddp.contrast:
    #     params = [p for p in model_without_ddp.contrast.parameters() if p.requires_grad]
    #     params_to_optimize.append({"params": params, "lr": args.lr * 10})

    # aux
    if args.aux:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr}) 
            
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    return optimizer



# def optim_manage(args, model_without_ddp):

#     params_to_optimize = [
#         {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
#         {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
#     ]
#     if args.aux:
#         params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
#         params_to_optimize.append({"params": params, "lr": args.lr * 10}) 
#     if args.contrast != -1:
#         if args.loss_name in ['simsiam', 'aspp_loss']:
#             if  'mep' not in args.model_name:
#                 params_simsiam = [p for p in model_without_ddp.contrast.parameters() if p.requires_grad]
#                 params_to_optimize.append({"params": params_simsiam, "lr": args.lr * 10})
#                 if args.attention:
#                     attention = [p for p in model_without_ddp.attention.parameters() if p.requires_grad]
#                     params_to_optimize.append({"params": attention, "lr": args.lr * 10})
#         elif args.loss_name == "intra":
#             if args.L3_loss != 0:
#                 params_L3u = [p for p in model_without_ddp.ProjectorHead_3u.parameters() if p.requires_grad]
#                 params_to_optimize.append({"params": params_L3u, "lr": args.lr * 10})
#             if args.L2_loss != 0:
#                 params_L2u = [p for p in model_without_ddp.ProjectorHead_2u.parameters() if p.requires_grad]
#                 params_to_optimize.append({"params": params_L2u, "lr": args.lr * 10})
#             if args.L1_loss != 0:
#                 params_L1u = [p for p in model_without_ddp.ProjectorHead_1u.parameters() if p.requires_grad]
#                 params_to_optimize.append({"params": params_L1u, "lr": args.lr * 10})
#         else:
#             if args.L3_loss != 0:
#                 params_L3d = [p for p in model_without_ddp.ProjectorHead_3d.parameters() if p.requires_grad]
#                 params_L3u = [p for p in model_without_ddp.ProjectorHead_3u.parameters() if p.requires_grad]
#                 params_to_optimize.append({"params": params_L3d, "lr": args.lr * 10})
#                 params_to_optimize.append({"params": params_L3u, "lr": args.lr * 10})
#             if args.L2_loss != 0:
#                 params_L2d = [p for p in model_without_ddp.ProjectorHead_2d.parameters() if p.requires_grad]
#                 params_L2u = [p for p in model_without_ddp.ProjectorHead_2u.parameters() if p.requires_grad]
#                 params_to_optimize.append({"params": params_L2d, "lr": args.lr * 10})
#                 params_to_optimize.append({"params": params_L2u, "lr": args.lr * 10})
#             if args.L1_loss != 0:
#                 params_L1d = [p for p in model_without_ddp.ProjectorHead_1d.parameters() if p.requires_grad]
#                 params_L1u = [p for p in model_without_ddp.ProjectorHead_1u.parameters() if p.requires_grad]
#                 params_to_optimize.append({"params": params_L1d, "lr": args.lr * 10})
#                 params_to_optimize.append({"params": params_L1u, "lr": args.lr * 10})
            
#     optimizer = torch.optim.SGD(
#         params_to_optimize,
#         lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
#     return optimizer