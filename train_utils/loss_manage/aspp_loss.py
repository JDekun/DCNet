import torch
import torch.nn as nn
from .SamplesModel import Sampling

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

def Contrastive(feats_x, feats_y, labels_, queue=None, queue_label=None, type: str = 'intra', temperature: float = 0.1, base_temperature: float = 0.07):
    anchor_num, n_view = feats_x.shape[0], feats_x.shape[1]

    feature_x = torch.cat(torch.unbind(feats_x, dim=1), dim=0)
    feature_y = torch.cat(torch.unbind(feats_y, dim=1), dim=0)

    # 默认采用 type == "double" 对比
    anchor_feature = torch.cat([feature_x, feature_y], dim=0)
    contrast_feature = anchor_feature
    anchor_count = n_view * 2
    contrast_count = n_view * 2

    if type == "inter":
        anchor_feature = feature_x
        contrast_feature= feature_y
        anchor_count = n_view
        contrast_count = n_view
    elif type == "intra":
        anchor_feature = feature_x
        contrast_feature= feature_x
        anchor_count = n_view
        contrast_count = n_view
         
    # 基础mask
    labels_ = labels_.contiguous().view(-1, 1)
    labels_T = labels_
    mask = torch.eq(labels_, torch.transpose(labels_T, 0, 1)).float().cuda()
    mask = mask.repeat(anchor_count, contrast_count)
    
    if queue is not None:
        queue_feature, queue_label = sample_negative(queue, queue_label) # 并行队列变形成串行

        # 增加queue特征
        contrast_feature = torch.cat([contrast_feature, queue_feature], dim=0)

        # 增加queue mask
        queue_label = queue_label.contiguous().view(-1, 1)
        mask_queue = torch.eq(labels_, torch.transpose(queue_label, 0, 1)).float().cuda()
        mask_queue = mask_queue.repeat(anchor_count, 1)
        # 更新mask
        mask = torch.cat([mask, mask_queue], dim=1)


    # 计算对比logits
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # logits = anchor_dot_contrast

    # mask对角线logits(自身对比部分)
    logits_mask = torch.ones_like(mask).scatter_(1,
                                                torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                0)
    # 正样本mask
    ops_mask = mask * logits_mask
    if type == "inter":
        ops_mask = mask
    # 负样本mask
    neg_mask = 1 - mask

    # 负样本对比总和
    exp_logits = torch.exp(logits)
    neg_logits = exp_logits * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    # 防止出现都正样本个数为0的情况
    ops_mask_num = ops_mask.sum(1)
    for i in range(len(ops_mask_num)):
        if ops_mask_num[i] == 0:
            ops_mask_num[i] = 1 

    # 计算对比损失
    log_prob = logits - torch.log(exp_logits + neg_logits)
    mean_log_prob_pos = (ops_mask * log_prob).sum(1) / ops_mask_num
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()

    return loss


def ASPP_CONTRAST_Loss(args, epoch, epochs, x, labels=None, predict=None):
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
        queue_origin = x[2]
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

    type = args.sample
    feats_, feats_y_, labels_, feats_que_, feats_y_que_, labels_queue_ = Sampling(type, epoch, epochs, feats, feats_y, labels, predict)
    # feats_, feats_y_, labels_ = Random_sampling(feats, feats_y, labels, predict)

    if feats_ != None:
        loss = Contrastive(feats_, feats_y_, labels_, queue, queue_label)
        if args.memory_size:
            dequeue_and_enqueue_self_seri(args, feats_que_, feats_y_que_, labels_queue_,
                                            encode_queue=queue_origin['encode_queue'],
                                            encode_queue_ptr=queue_origin['encode_queue_ptr'],
                                            code_queue_label=queue_origin['code_queue_label'])
    else:
        loss = 0

    return loss