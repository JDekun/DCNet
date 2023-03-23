import torch
import torch.nn as nn


def self_pace3(epoch, epochs, num_easy, num_hard):

    archor = epochs//3

    if  archor > epoch:
        num_hard_keep = 0
        num_easy_keep = num_easy
    elif 2*archor > epoch:
        num_hard_keep = num_hard
        num_easy_keep = num_easy
    else:
        num_easy_keep = 0
        num_hard_keep = num_hard
    
    return num_easy_keep, num_hard_keep

def only_esay(epoch, epochs, num_easy, num_hard):

    num_hard_keep = 0
    num_easy_keep = num_easy
    
    return num_easy_keep, num_hard_keep




def Self_pace3_concat_sampling(epoch, epochs, X, Y, y_hat, y, ignore_label: int = 255, max_views: int = 50, max_samples: int = 1024):
    batch_size, feat_dim = X.shape[0], X.shape[-1]
    classes = []
    total_classes = 0
    for ii in range(batch_size):
        this_y = y_hat[ii]
        this_classes = torch.unique(this_y)
        this_classes = [x for x in this_classes if x != ignore_label]
        # this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > max_views]

        classes.append(this_classes)
        total_classes += len(this_classes)

    if total_classes == 0:
        return None, None

    # n_view = max_samples // total_classes
    # n_view = min(n_view, max_views)

    X_ = torch.zeros((total_classes, 1, feat_dim), dtype=torch.float).cuda()
    Y_ = torch.zeros((total_classes, 1, feat_dim), dtype=torch.float).cuda()
    y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
    

    X_ptr = 0
    for ii in range(batch_size):
        this_y_hat = y_hat[ii]
        this_y = y[ii]
        this_classes = classes[ii]

        for cls_id in this_classes:
            hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
            easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

            num_hard = hard_indices.shape[0]
            num_easy = easy_indices.shape[0]

            archor = epochs//3
            if  archor > epoch:
                num_hard_keep = 0
                num_easy_keep = num_easy
            elif 2*archor > epoch:
                num_hard_keep = num_hard
                num_easy_keep = num_easy
            else:
                num_easy_keep = 0
                num_hard_keep = num_hard

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

def Self_pace3_sampling(epoch, epochs, X, Y, y_hat, y, ignore_label: int = 255, max_views: int = 50, max_samples: int = 1024):
    batch_size, feat_dim = X.shape[0], X.shape[-1]

    classes = []
    total_classes = 0
    for ii in range(batch_size):
        this_y = y_hat[ii]
        this_classes = torch.unique(this_y)
        this_classes = [x for x in this_classes if x != ignore_label]
        this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > max_views]

        classes.append(this_classes)
        total_classes += len(this_classes)

    if total_classes == 0:
        return None, None

    n_view = max_samples // total_classes
    n_view = min(n_view, max_views)

    X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
    Y_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
    y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
    
    X_ptr = 0
    for ii in range(batch_size):
        this_y_hat = y_hat[ii]
        this_y = y[ii]
        this_classes = classes[ii]

        for cls_id in this_classes:
            hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
            easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

            num_hard = hard_indices.shape[0]
            num_easy = easy_indices.shape[0]

            archor = epochs//3
            if  archor > epoch:
                num_hard_keep = 0
                if num_easy > n_view:
                    num_easy_keep = n_view
                else:
                    num_easy_keep = num_easy
            elif 2*archor > epoch:
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception
            else:
                num_easy_keep = 0
                if num_hard > n_view:
                    num_hard_keep = n_view
                else:
                    num_hard_keep = num_hard

            perm = torch.randperm(num_hard)
            hard_indices = hard_indices[perm[:num_hard_keep]]
            perm = torch.randperm(num_easy)
            easy_indices = easy_indices[perm[:num_easy_keep]] 
            indices = torch.cat((hard_indices, easy_indices), dim=0)

            temp = indices.shape[0]
            if temp != 0:
                X_[X_ptr, 0:temp, :] = X[ii, indices, :].squeeze(1)
                Y_[X_ptr, 0:temp, :] = Y[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

    return X_, Y_, y_, X_.detach(), X_.detach(), y_.detach()