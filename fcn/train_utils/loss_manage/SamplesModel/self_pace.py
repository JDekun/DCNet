
def self_pace3(epoch, epochs, num_easy, num_hard):

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
    
    return num_easy_keep, num_hard_keep

def only_esay(epoch, epochs, num_easy, num_hard):

    num_hard_keep = 0
    num_easy_keep = num_easy
    
    return num_easy_keep, num_hard_keep