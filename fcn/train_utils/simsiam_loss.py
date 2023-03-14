import torch
import torch.nn as nn

def Main_sampling(X, Y, y_hat, ignore_label: int = 255):
    batch_size = X.shape[0]

    X = X.permute(0, 2, 3, 1)
    Y = Y.permute(0, 2, 3, 1)

    ii = 0
    this_y_hat = y_hat[0]
    indices = (this_y_hat != ignore_label).nonzero()
    X = X[ii, indices]

    # X_ = X[ii, indices, :]
    # Y_ = Y[ii, indices, :]

    print(X.shape)
    # print(X_.shape)

    for ii in range(batch_size-1):
        ii = ii + 1
        this_y_hat = y_hat[ii]

        indices = (this_y_hat != ignore_label).nonzero()

        SAM_X = X[ii, indices, :].squeeze(1)
        SAM_Y = Y[ii, indices, :].squeeze(1)

        X_ = torch.cat(X_, SAM_X, 0)
        Y_ = torch.cat(Y_, SAM_Y, 0)

    return X_, Y_

def simsiam_loss(criterion, conen, conde, target, ignore_index=255):
    conen, conde = Main_sampling(conen, conde, target, ignore_index)
    
    loss = -criterion(conen, conde).mean() * 0.5

    return loss