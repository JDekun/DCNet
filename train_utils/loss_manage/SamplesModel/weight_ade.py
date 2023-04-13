import torch

def weight_ade(this_y_hat, this_y, cls_id, ade):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]
    sum = num_hard + num_easy

    # >>>> 核心修改部分
    n = ade
    easy_rate = (num_easy/sum) # ↑
    hard_rate = (num_hard/sum) # ↓
    print('easy_rate',easy_rate)
    print('hard_rate',hard_rate)

    hard = (hard_rate + 1)**(-n)  # y= (x+1)^-8
    easy = (easy_rate + 1)**(-n)  # y= (x+1)^-8

    # <<<<
    
    
    return hard_indices, easy_indices, hard, easy
