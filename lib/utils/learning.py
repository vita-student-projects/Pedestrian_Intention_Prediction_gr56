import os
import numpy as np
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

def load_pretrained_weights(model, checkpoint):
    """Load pretrained weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model

def partial_train_layers(model, partial_list):
    """Train partial layers of a given model."""
    for name, p in model.named_parameters():
        p.requires_grad = False
        for trainable in partial_list:
            if trainable in name:
                p.requires_grad = True
                break
    return model

def load_backbone(args):
    """Load backbone model."""
    model_backbone = DSTformer(dim_in=args.dim_bbox, dim_out=args.dim_bbox, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                               depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                               maxlen=args.maxlen, num_joints=args.num_joints)
    
    return model_backbone