import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from lib.model.DSTformer import DSTformer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target):
    """Computes the accuracy over the predictions"""
    with torch.no_grad():
        pred = F.softmax(output, dim=1)
        equality = (target == pred.argmax(dim=1))
        accuracy = equality.type_as(torch.FloatTensor()).mean()
    return accuracy

def load_backbone(args):
    """Load backbone model."""
    model_backbone = DSTformer(dim_in=args.dim_in, dim_out=args.dim_out, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                               depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                               maxlen=args.maxlen, num_joints=args.num_joints)
    
    return model_backbone