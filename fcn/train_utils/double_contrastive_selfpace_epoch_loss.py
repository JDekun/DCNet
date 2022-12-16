import torch
import torch.nn as nn

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

    return X_, Y_, y_

def Self_pace2_sampling(epoch, epochs, X, Y, y_hat, y, ignore_label: int = 255, max_views: int = 50, max_samples: int = 1024):
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

            archor = epochs//2
            if  archor > epoch:
                num_hard_keep = 0
                if num_easy > n_view:
                    num_easy_keep = n_view
                else:
                    num_easy_keep = num_easy
            else:
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

            temp = indices.shape[0]
            if temp != 0:
                X_[X_ptr, 0:temp, :] = X[ii, indices, :].squeeze(1)
                Y_[X_ptr, 0:temp, :] = Y[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

    return X_, Y_, y_

def Random_sampling(X, Y, y_hat, y, ignore_label: int = 255, max_views: int = 50, max_samples: int = 1024):
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
        this_classes = classes[ii]

        for cls_id in this_classes:
            hard_indices = (this_y_hat == cls_id).nonzero()

            num_hard = hard_indices.shape[0]

            perm = torch.randperm(num_hard)
            hard_indices = hard_indices[perm[:n_view]]
            indices = hard_indices

            X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
            Y_[X_ptr, :, :] = Y[ii, indices, :].squeeze(1)
            y_[X_ptr] = cls_id
            X_ptr += 1

    return X_, Y_, y_

def sample_negative(Q):
    class_num, cache_size, feat_size = Q.shape

    X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
    y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
    sample_ptr = 0
    for ii in range(class_num):
        if ii == 0: continue
        this_q = Q[ii, :cache_size, :]

        X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
        y_[sample_ptr:sample_ptr + cache_size, ...] = ii
        sample_ptr += cache_size

    return X_, y_

def Contrastive(feats_, feats_y_, labels_, queue=None, temperature: float = 0.1, base_temperature: float = 0.07):
    anchor_num, n_view = feats_.shape[0], feats_.shape[1]

    labels_ = labels_.contiguous().view(-1, 1)
    contrast_feature_y = torch.cat(torch.unbind(feats_y_, dim=1), dim=0)

    if queue is not None:
        X_contrast, y_contrast = sample_negative(queue)
        y_contrast = y_contrast.contiguous().view(-1, 1)
        contrast_count = 1
        contrast_feature = X_contrast
    else:
        y_contrast = labels_
        contrast_count = n_view * 2
        contrast_feature_x = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        contrast_feature = torch.cat([contrast_feature_y, contrast_feature_x], dim=0)

    # 1*N
    anchor_feature = contrast_feature_y
    anchor_count = n_view
    # n*n
    # anchor_feature = contrast_feature
    # anchor_count = contrast_count

    mask = torch.eq(labels_, torch.transpose(y_contrast, 0, 1)).float().cuda()

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

def dequeue_and_enqueue(args, keys, key_y, labels,
                        encode_queue, encode_queue_ptr,
                        decode_queue, decode_queue_ptr):
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]

    labels = labels[:, ::args.network_stride, ::args.network_stride]

    for bs in range(batch_size):
        this_feat = keys[bs].contiguous().view(feat_dim, -1)
        this_feat_y = key_y[bs].contiguous().view(feat_dim, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0 and x != 255]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero()

            # segment enqueue and dequeue
            # feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
            # ptr = int(encode_queue_ptr[lb])
            # encode_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
            # encode_queue_ptr[lb] = (encode_queue_ptr[lb] + 1) % args.memory_size

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, args.pixel_update_freq)
            # feat = feat[:, perm[:K]]
            feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)
            feat_y = this_feat_y[:, perm[:K]]
            feat_y = torch.transpose(feat_y, 0, 1)
            ptr = int(decode_queue_ptr[lb])

            if ptr + K >= args.memory_size:
                decode_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                decode_queue_ptr[lb] = 0
                encode_queue[lb, -K:, :] = nn.functional.normalize(feat_y, p=2, dim=1)
                encode_queue[lb] = 0
            else:
                decode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                decode_queue_ptr[lb] = (decode_queue_ptr[lb] + 1) % args.memory_size
                encode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat_y, p=2, dim=1)
                encode_queue_ptr[lb] = (encode_queue_ptr[lb] + 1) % args.memory_size


def EPOCHSELFPACEDoublePixelContrastLoss(args, epoch, epochs, x, labels=None, predict=None):
    feats = x[0]
    feats_y = x[1]

    labels = labels.unsqueeze(1).float().clone()
    labels = torch.nn.functional.interpolate(labels,
                                                (feats.shape[2], feats.shape[3]), mode='nearest')
    labels = labels.squeeze(1).long()
    assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

    queue=None
    if args.memory_size:
        queue = x[5]
        # 更新队列
        feats_que = x[2]
        feats_y_que = x[3]
        labels_que = x[4]
        dequeue_and_enqueue(args, feats_que, feats_y_que, labels_que,
                            encode_queue=queue["encode_queue"],
                            encode_queue_ptr=queue["encode_queue_ptr"],
                            decode_queue=queue["decode_queue"],
                            decode_queue_ptr=queue["decode_queue_ptr"])

        if "encode_queue" in queue:
            encode_queue = queue['encode_queue']
        else:
            encode_queue = None

        if "decode_queue" in queue:
            decode_queue = queue['decode_queue']
        else:
            decode_queue = None

    if encode_queue is not None and decode_queue is not None:
        queue = torch.cat((encode_queue, decode_queue), dim=1)

    batch_size = feats.shape[0]

    labels = labels.contiguous().view(batch_size, -1)
    predict = predict.contiguous().view(batch_size, -1)

    feats = feats.permute(0, 2, 3, 1)
    feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
    feats_y = feats_y.permute(0, 2, 3, 1)
    feats_y = feats_y.contiguous().view(feats_y.shape[0], -1, feats_y.shape[-1])

    feats_, feats_y_, labels_ = Self_pace3_sampling(epoch, epochs, feats, feats_y, labels, predict)
    # feats_, feats_y_, labels_ = Random_sampling(feats, feats_y, labels, predict)

    loss = Contrastive(feats_, feats_y_, labels_, queue)
    return loss