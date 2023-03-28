import torch

def self_pace_epochs(epoch, epochs, this_y_hat, this_y, cls_id):
    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
    num_hard = hard_indices.shape[0]
    num_easy = easy_indices.shape[0]

    # >>>>
    rate_hard = (epoch/epochs)
    rate_easy_threshold = (1 - rate_hard) if (1 - rate_hard) > 1/4 else 1/4
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
