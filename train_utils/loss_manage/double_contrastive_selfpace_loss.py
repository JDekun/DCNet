import torch


def Hard_anchor_sampling(X, Y, y_hat, y, ignore_label: int = 255, max_views: int = 50, max_samples: int = 1024):
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
            archor = (num_hard + num_easy)//3

            if  archor > num_easy:
                num_hard_keep = 0
                if num_easy > n_view:
                    num_easy_keep = n_view
                else:
                    num_easy_keep = num_easy
            elif 2*archor > num_easy:
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
                    num_easy_keep = n_view
                else:
                    num_easy_keep = num_hard

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
    anchor_num, n_view = feats_.shape[0], feats_.shape[1]

    labels_ = labels_.contiguous().view(-1, 1)
    mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

    contrast_count = n_view * 2
    contrast_feature_x = torch.cat(torch.unbind(feats_, dim=1), dim=0)
    contrast_feature_y = torch.cat(torch.unbind(feats_y_, dim=1), dim=0)
    contrast_feature = torch.cat([contrast_feature_x, contrast_feature_y], dim=0)

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                    temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    mask = mask.repeat(anchor_count, contrast_count)
    neg_mask = 1 - mask

    logits_mask = torch.ones_like(mask).scatter_(1,
                                                torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                0)
    mask = mask * logits_mask

    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    exp_logits = torch.exp(logits)

    log_prob = logits - torch.log(exp_logits + neg_logits)

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()

    return loss


def SELFPACEDoublePixelContrastLoss(x, labels=None, predict=None):
    feats = x[0]
    feats_y = x[1]
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