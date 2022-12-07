import torch


def Hard_anchor_sampling(X, Y, y_hat, y, ignore_label: int = 255, max_views: int = 100, max_samples: int = 1024):
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

            perm = torch.randperm(num_hard)
            hard_indices = hard_indices[perm[:num_hard_keep]]
            perm = torch.randperm(num_easy)
            easy_indices = easy_indices[perm[:num_easy_keep]]
            indices = torch.cat((hard_indices, easy_indices), dim=0)

            X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
            Y_[X_ptr, :, :] = Y[ii, indices, :].squeeze(1)
            y_[X_ptr] = cls_id
            X_ptr += 1

    return X_, Y_, y_

def Contrastive(feats_, feats_y_, labels_, temperature: float = 0.1, base_temperature: float = 0.07):
    batch_size = feats_.shape[0] * feats_.shape[1]

    
    out_1 = feats_.contiguous().view([-1, feats_.shape[-1]])
    out_2 = feats_y_.contiguous().view([-1, feats_y_.shape[-1]])
    # [2*B*H*W, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B*H*W, 2*B*H*W]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B*H*W, 2*B*H*W-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.sum(out_1 * out_2, dim=-1) / temperature
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (pos_sim - torch.log(sim_matrix.sum(dim=-1))).mean()

    loss = - (temperature / base_temperature) * loss
    loss = loss.mean()

    return loss

def InterPixelContrastLoss(feats, feats_y=None, labels=None, predict=None):
    labels = labels.unsqueeze(1).float().clone()
    labels = torch.nn.functional.interpolate(labels,
                                                (feats.shape[2], feats.shape[3]), mode='nearest')
    labels = labels.squeeze(1).long()
    assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

    batch_size = feats.shape[0]

    labels = labels.contiguous().view(batch_size, -1)
    predict = predict.contiguous().view(batch_size, -1)

    feats = feats.permute(0, 2, 3, 1)
    feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
    feats_y = feats_y.permute(0, 2, 3, 1)
    feats_y = feats_y.contiguous().view(feats_y.shape[0], -1, feats_y.shape[-1])

    feats_, feats_y_, labels_ = Hard_anchor_sampling(feats, feats_y, labels, predict)

    loss = Contrastive(feats_, feats_y_, labels_)
    return loss