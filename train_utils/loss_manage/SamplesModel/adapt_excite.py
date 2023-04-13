import torch

def adapt_excite(this_y_hat, this_y, cls_id, aex):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]
    sum = num_hard + num_easy

    # >>>> 核心修改部分
    n = aex
    easy_rate = (num_easy/sum) # ↑
    hard_rate = (num_hard/sum) # ↓

    easy = (easy_rate + 1)**(-n)  # y= (x+1)^-2
    hard = (hard_rate + 1)**(-n)  # y= (x+1)^-2

    num_hard_keep = round(num_hard * hard)
    num_easy_keep = round(num_easy * easy)
    # <<<<

    perm = torch.randperm(num_hard)
    hard_indices = hard_indices[perm[:num_hard_keep]]
    perm = torch.randperm(num_easy)
    easy_indices = easy_indices[perm[:num_easy_keep]] 
    indices = torch.cat((hard_indices, easy_indices), dim=0)
    
    return indices
