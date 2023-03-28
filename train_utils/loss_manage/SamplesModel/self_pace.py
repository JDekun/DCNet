
def self_pace_epochs(epoch, epochs, num_easy, num_hard):

    rate_hard = (epoch/epochs)
    rate_easy_threshold = (1 - rate_hard) if (1 - rate_hard) > 1/4 else 1/4

    num_hard_keep = round(rate_hard * num_hard)
    num_easy_keep = round(rate_easy_threshold * num_easy)
    
    return num_easy_keep, num_hard_keep

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
