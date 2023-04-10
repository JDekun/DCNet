import torch

def self_pace_ploy(epoch, epochs, this_y_hat, this_y, cls_id):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]
    sum = num_hard + num_easy

    # >>>> 核心修改部分
    n = 8
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

def self_pace_step(epoch, epochs, this_y_hat, this_y, cls_id):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]

    # >>>> 核心修改部分
    n = 2
    step = (epoch/epochs)
    rate_hard = step**(1/n) # y= x^0.5
    rate_easy = (step + 1)**(-n)  # y= (x+1)^-2
 
    num_hard_keep = round(rate_hard * num_hard)
    num_easy_keep = round(rate_easy * num_easy)
    # <<<<

    perm = torch.randperm(num_hard)
    hard_indices = hard_indices[perm[:num_hard_keep]]
    perm = torch.randperm(num_easy)
    easy_indices = easy_indices[perm[:num_easy_keep]] 
    indices = torch.cat((hard_indices, easy_indices), dim=0)
    
    return indices

def self_pace_epochs(epoch, epochs, this_y_hat, this_y, cls_id):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]

    # >>>> 核心修改部分
    rate_hard = (epoch/epochs)
    easy_hard = 1 - rate_hard
    rate_easy_threshold = easy_hard if easy_hard > 1/4 else 1/4
    num_hard_keep = round(rate_hard * num_hard)
    num_easy_keep = round(rate_easy_threshold * num_easy)
    # <<<<

    perm = torch.randperm(num_hard)
    hard_indices = hard_indices[perm[:num_hard_keep]]
    perm = torch.randperm(num_easy)
    easy_indices = easy_indices[perm[:num_easy_keep]] 
    indices = torch.cat((hard_indices, easy_indices), dim=0)
    
    return indices

def self_pace3(epoch, epochs, this_y_hat, this_y, cls_id):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]

    # >>>>
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
    # <<<<

    perm = torch.randperm(num_hard)
    hard_indices = hard_indices[perm[:num_hard_keep]]
    perm = torch.randperm(num_easy)
    easy_indices = easy_indices[perm[:num_easy_keep]] 
    indices = torch.cat((hard_indices, easy_indices), dim=0)
    
    return indices
