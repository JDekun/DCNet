import torch
import torch.nn as nn
from train_utils.loss_manage import SamplesModel

def Sampling(type, epoch, epochs, X, Y, labels, predict, ignore_label: int = 255):
    batch_size, feat_dim = X.shape[0], X.shape[-1]

    classes = []
    total_classes = 0
    for ii in range(batch_size):
        this_y = labels[ii]
        this_classes = torch.unique(this_y)
        this_classes = [x for x in this_classes if x != ignore_label]
        classes.append(this_classes)
        total_classes += len(this_classes)

    if total_classes == 0:
        return None, None, None, None, None, None

    X_ = torch.zeros((total_classes, 1, feat_dim), dtype=torch.float).cuda()
    Y_ = torch.zeros((total_classes, 1, feat_dim), dtype=torch.float).cuda()
    y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
    
    X_ptr = 0
    for ii in range(batch_size):
        this_y_hat = labels[ii]
        this_y = predict[ii]
        this_classes = classes[ii]

        for cls_id in this_classes:
            if "weight_ade" in type:
                w = int(type.split('_')[-1])
                hard_indices, easy_indices , hard_weight, easy_weight = eval("SamplesModel." + "weight_ade")(this_y_hat, this_y, cls_id, w)
                num_h = hard_indices.shape[0]
                num_e = easy_indices.shape[0]
                if num_e == 0:
                    ade_x = torch.mean(X[ii, hard_indices, :].squeeze(1), dim=0)*hard_weight
                    ade_y = torch.mean(Y[ii, hard_indices, :].squeeze(1), dim=0)*hard_weight
                elif num_h == 0:
                    ade_x = torch.mean(X[ii, easy_indices, :].squeeze(1), dim=0)*easy_weight
                    ade_y = torch.mean(Y[ii, easy_indices, :].squeeze(1), dim=0)*easy_weight
                else:
                    ade_x = torch.mean(X[ii, hard_indices, :].squeeze(1), dim=0)*hard_weight  + torch.mean(X[ii, easy_indices, :].squeeze(1), dim=0)*easy_weight
                    ade_y = torch.mean(Y[ii, hard_indices, :].squeeze(1), dim=0)*hard_weight  + torch.mean(Y[ii, easy_indices, :].squeeze(1), dim=0)*easy_weight
                X_[X_ptr, 0, :] = ade_x
                Y_[X_ptr, 0, :] = ade_y
                y_[X_ptr] = cls_id
            else:  
                if "adapt_excite" in type:
                    n = int(type.split('_')[-1])
                    indices = eval("SamplesModel." + "adapt_excite")(this_y_hat, this_y, cls_id, n)
                elif "self_pace" in type:
                    indices = eval("SamplesModel." + type)(epoch, epochs, this_y_hat, this_y, cls_id)
                else:
                    indices = eval("SamplesModel." + type)(this_y_hat, this_y, cls_id)

                temp = indices.shape[0]
                if temp != 0:
                    X_[X_ptr, 0, :] = torch.mean(X[ii, indices, :].squeeze(1), dim=0)
                    Y_[X_ptr, 0, :] = torch.mean(Y[ii, indices, :].squeeze(1), dim=0)
                    y_[X_ptr] = cls_id
                X_ptr += 1

    return X_, Y_, y_, X_.detach(), Y_.detach(), y_.detach()
