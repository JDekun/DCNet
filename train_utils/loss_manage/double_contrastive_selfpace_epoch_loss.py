import torch
import torch.nn as nn

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

def sample_negative(Q, Q_label):
    class_num, cache_size, feat_size = Q.shape

    X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
    y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
    sample_ptr = 0
    for ii in range(class_num):
        # if ii == 0: continue
        this_q = Q[ii, :cache_size, :]
        # this_q_label = Q_label[ii, :cache_size]

        X_[sample_ptr:sample_ptr + cache_size, :] = this_q
        y_[sample_ptr:sample_ptr + cache_size, :] = torch.transpose(Q_label, 0, 1)
        sample_ptr += cache_size

    return X_, y_

def dequeue_and_enqueue(args, keys, key_y, labels,
                        encode_queue, encode_queue_ptr,
                        decode_queue, decode_queue_ptr):
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]

    labels = labels[:, ::args.network_stride, ::args.network_stride]
    memory_size = args.memory_size

    for bs in range(batch_size):
        this_feat = keys[bs].contiguous().view(feat_dim, -1)
        this_feat_y = key_y[bs].contiguous().view(feat_dim, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x != 255]

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

            if ptr + K > memory_size:
                total = ptr + K
                start = total - memory_size
                end = K - start

                encode_queue[lb, ptr:memory_size, :] = nn.functional.normalize(feat[0:end], p=2, dim=1)
                encode_queue[lb, 0:start, :] = nn.functional.normalize(feat[end:], p=2, dim=1)
                encode_queue_ptr[lb] = start
                decode_queue[lb, ptr:memory_size, :] = nn.functional.normalize(feat_y[0:end], p=2, dim=1)
                decode_queue[lb, 0:start, :] = nn.functional.normalize(feat_y[end:], p=2, dim=1)
                decode_queue_ptr[lb] = start

                # encode_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                # encode_queue_ptr[lb] = 0
                # decode_queue[lb, -K:, :] = nn.functional.normalize(feat_y, p=2, dim=1)
                # decode_queue_ptr[lb] = 0
            else:
                encode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                encode_queue_ptr[lb] = (encode_queue_ptr[lb] + K) % args.memory_size
                decode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat_y, p=2, dim=1)
                decode_queue_ptr[lb] = (decode_queue_ptr[lb] + K) % args.memory_size

def dequeue_and_enqueue_self(args, keys, key_y, labels,
                            encode_queue, encode_queue_ptr,
                            decode_queue, decode_queue_ptr):
    memory_size = args.memory_size

    iter =  len(labels)
    for i in range(iter):
        lb = int(labels[i])
        feat = keys[i]
        feat_y = key_y[i]
        K = feat.shape[0]

        ptr = int(decode_queue_ptr[lb])

        if ptr + K > memory_size:
            total = ptr + K
            start = total - memory_size
            end = K - start

            encode_queue[lb, ptr:memory_size, :] = nn.functional.normalize(feat[0:end], p=2, dim=1)
            encode_queue[lb, 0:start, :] = nn.functional.normalize(feat[end:], p=2, dim=1)
            encode_queue_ptr[lb] = start
            decode_queue[lb, ptr:memory_size, :] = nn.functional.normalize(feat_y[0:end], p=2, dim=1)
            decode_queue[lb, 0:start, :] = nn.functional.normalize(feat_y[end:], p=2, dim=1)
            decode_queue_ptr[lb] = start

        else:
            encode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
            encode_queue_ptr[lb] = (encode_queue_ptr[lb] + K) % args.memory_size
            decode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat_y, p=2, dim=1)
            decode_queue_ptr[lb] = (decode_queue_ptr[lb] + K) % args.memory_size

def dequeue_and_enqueue_self_seri(args, keys, key_y, labels,
                                encode_queue, encode_queue_ptr,
                                code_queue_label):
    memory_size = args.memory_size

    iter =  len(labels)
    for i in range(iter):
        lb = 0
        lbe = int(labels[i])
        feat = keys[i]
        feat_y = key_y[i]
        K = feat.shape[0]

        ptr = int(encode_queue_ptr[lb])

        if ptr + K > memory_size:
            total = ptr + K
            start = total - memory_size
            end = K - start

            encode_queue[lb, ptr:memory_size, :] = nn.functional.normalize(feat[0:end], p=2, dim=1)
            encode_queue[lb, 0:start, :] = nn.functional.normalize(feat[end:], p=2, dim=1)
            encode_queue_ptr[lb] = start
            # decode_queue[lb, ptr:memory_size, :] = nn.functional.normalize(feat_y[0:end], p=2, dim=1)
            # decode_queue[lb, 0:start, :] = nn.functional.normalize(feat_y[end:], p=2, dim=1)
            # decode_queue_ptr[lb] = start

            code_queue_label[lb, ptr:memory_size] = lbe
            code_queue_label[lb, 0:start] = lbe

        else:
            encode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
            encode_queue_ptr[lb] = (encode_queue_ptr[lb] + K) % args.memory_size
            # decode_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat_y, p=2, dim=1)
            # decode_queue_ptr[lb] = (decode_queue_ptr[lb] + K) % args.memory_size

            code_queue_label[lb, ptr:ptr + K] = lbe

def Contrastive(feats_, feats_y_, labels_, queue=None, queue_label=None, temperature: float = 0.1, base_temperature: float = 0.07):
    anchor_num, n_view = feats_.shape[0], feats_.shape[1]

    labels_ = labels_.contiguous().view(-1, 1)
    contrast_feature_y = torch.cat(torch.unbind(feats_y_, dim=1), dim=0)

    # 1*N
    anchor_feature = contrast_feature_y
    anchor_count = n_view
    # n*n
    # anchor_feature = contrast_feature
    # anchor_count = contrast_count
  
    y_contrast = labels_
    contrast_count = n_view
    contrast_feature_x = torch.cat(torch.unbind(feats_, dim=1), dim=0)
    # contrast_feature = torch.cat([contrast_feature_y, contrast_feature_x], dim=0)
    contrast_feature = contrast_feature_x

    mask = torch.eq(labels_, torch.transpose(y_contrast, 0, 1)).float().cuda()
    
    mask = mask.repeat(anchor_count, contrast_count)
    

    if queue is not None:
        X_contrast, y_contrast_queue = sample_negative(queue, queue_label) # 并行队列变形成串行

        y_contrast_queue = y_contrast_queue.contiguous().view(-1, 1)
        contrast_count_queue = 1
        contrast_feature = torch.cat([contrast_feature, X_contrast], dim=0)
        # contrast_feature = X_contrast

        mask_queue = torch.eq(labels_, torch.transpose(y_contrast_queue, 0, 1)).float().cuda()
        mask_queue = mask_queue.repeat(anchor_count, contrast_count_queue)

        mask = torch.cat([mask, mask_queue], dim=1)
        # mask = mask_queue


    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                    temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.ones_like(mask).scatter_(1,
                                                torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                0)
    
    ops_mask = mask * logits_mask
    neg_mask = 1 - mask
    
    exp_logits = torch.exp(logits)

    neg_logits = exp_logits * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)
    

    log_prob = logits - torch.log(exp_logits + neg_logits)

    ops_mask_num = ops_mask.sum(1)
    for i in range(len(ops_mask_num)):
        if ops_mask_num[i] == 0:
            ops_mask_num[i] = 1 

    mean_log_prob_pos = (ops_mask * log_prob).sum(1) / ops_mask_num

    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()

    return loss


def EPOCHSELFPACEDoublePixelContrastLoss(args, epoch, epochs, x, labels=None, predict=None):
    feats = x[0]
    feats_y = x[1]

    labels = labels.unsqueeze(1).float().clone()
    labels = torch.nn.functional.interpolate(labels,
                                                (feats.shape[2], feats.shape[3]), mode='nearest')
    labels = labels.squeeze(1).long()
    assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

    queue=None
    queue_label=None
    if args.memory_size:
        queue_origin = x[5]
        # queue = queue_origin

        if "encode_queue" in queue_origin:
            encode_queue = queue_origin['encode_queue']
            encode_queue_label = queue_origin['code_queue_label']
        else:
            encode_queue = None

        # if "decode_queue" in queue:
        #     decode_queue = queue['decode_queue']
        #     decode_queue_label = queue['code_queue_label']
        # else:
        #     decode_queue = None

        # if encode_queue is not None and decode_queue is not None:
        #     queue = torch.cat((encode_queue, decode_queue), dim=1)
        #     queue_label = torch.cat((encode_queue_label, decode_queue_label), dim=1)
        queue = encode_queue
        queue_label = encode_queue_label

    batch_size = feats.shape[0]

    labels = labels.contiguous().view(batch_size, -1)
    predict = predict.contiguous().view(batch_size, -1)

    feats = feats.permute(0, 2, 3, 1)
    feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
    feats_y = feats_y.permute(0, 2, 3, 1)
    feats_y = feats_y.contiguous().view(feats_y.shape[0], -1, feats_y.shape[-1])

    feats_, feats_y_, labels_, feats_que_, feats_y_que_, labels_queue_ = Self_pace3_concat_sampling(epoch, epochs, feats, feats_y, labels, predict)
    # feats_, feats_y_, labels_ = Random_sampling(feats, feats_y, labels, predict)

    loss = Contrastive(feats_, feats_y_, labels_, queue, queue_label)

    # 并行更新队列
    # if args.memory_size:
    #     # dequeue_and_enqueue(args, feats_que, feats_y_que, labels_que,
    #     #                     encode_queue=queue_origin['encode_queue'],
    #     #                     encode_queue_ptr=queue_origin['encode_queue_ptr'],
    #     #                     decode_queue=queue_origin['decode_queue'],
    #     #                     decode_queue_ptr=queue_origin['decode_queue_ptr'])
    #     dequeue_and_enqueue_self(args, feats_que_, feats_y_que_, labels_queue_,
    #                                 encode_queue=queue_origin['encode_queue'],
    #                                 encode_queue_ptr=queue_origin['encode_queue_ptr'],
    #                                 decode_queue=queue_origin['decode_queue'],
    #                                 decode_queue_ptr=queue_origin['decode_queue_ptr'])

    if args.memory_size:
        # dequeue_and_enqueue(args, feats_que, feats_y_que, labels_que,
        #                     encode_queue=queue_origin['encode_queue'],
        #                     encode_queue_ptr=queue_origin['encode_queue_ptr'],
        #                     decode_queue=queue_origin['decode_queue'],
        #                     decode_queue_ptr=queue_origin['decode_queue_ptr'])
        dequeue_and_enqueue_self_seri(args, feats_que_, feats_y_que_, labels_queue_,
                                        encode_queue=queue_origin['encode_queue'],
                                        encode_queue_ptr=queue_origin['encode_queue_ptr'],
                                        code_queue_label=queue_origin['code_queue_label'])

    return loss