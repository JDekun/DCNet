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
        return None, None

    X_ = torch.zeros((total_classes, 1, feat_dim), dtype=torch.float).cuda()
    Y_ = torch.zeros((total_classes, 1, feat_dim), dtype=torch.float).cuda()
    y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
    
    X_ptr = 0
    for ii in range(batch_size):
        this_y_hat = labels[ii]
        this_y = predict[ii]
        this_classes = classes[ii]

        for cls_id in this_classes:
            hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
            easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

            num_hard = hard_indices.shape[0]
            num_easy = easy_indices.shape[0]

            num_easy_keep, num_hard_keep = eval("SamplesModel." + type)(epoch, epochs, num_easy, num_hard)

            perm = torch.randperm(num_hard)
            hard_indices = hard_indices[perm[:num_hard_keep]]
            perm = torch.randperm(num_easy)
            easy_indices = easy_indices[perm[:num_easy_keep]] 
            indices = torch.cat((hard_indices, easy_indices), dim=0)

            temp = indices.shape[0]
            if temp != 0:
                X_[X_ptr, 0, :] = torch.mean(X[ii, indices, :].squeeze(1), dim=0)
                Y_[X_ptr, 0, :] = torch.mean(Y[ii, indices, :].squeeze(1), dim=0)
                y_[X_ptr] = cls_id
                X_ptr += 1

    return X_, Y_, y_, X_.detach(), X_.detach(), y_.detach()
