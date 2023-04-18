import math

def weight_ade(this_y_hat, this_y, cls_id, ade):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]
    sum = num_hard + num_easy

    # >>>> 核心修改部分
    n = ade
    easy_rate = (num_easy/sum) # ↑
    hard_rate = 1-easy_rate # ↓

    hard = (hard_rate + 1)**(-n)  # y= (x+1)^-8
    easy = (easy_rate + 1)**(-n)  # y= (x+1)^-8

    # <<<<
    
    return hard_indices, easy_indices, hard, easy


def weight_ade_softmax(this_y_hat, this_y, cls_id, ade):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]
    sum = num_hard + num_easy

    # >>>> 核心修改部分
    n = ade
    easy_rate = (num_easy/sum) # ↑ 简单样本的比例在上升
    hard_weight = easy_rate # ↓ 难样本的权重在上升
    easy_weight = 1-easy_rate # ↓ 简单样本的权重在下降

    H = math.exp(hard_weight)
    E = math.exp(easy_weight)

    hard = H/(H+E)
    easy = E/(H+E)

    # <<<<
    
    return hard_indices, easy_indices, hard, easy