import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random
import pprint as pp # pretty print

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_jaad import JAADDataset
from lib.model.model_action import ActionNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/jaad/JAAD_train.yaml')
    parser.add_argument('-f', '--freq', type=int, default=100)
    opts = parser.parse_args()
    return opts

def train(args,opts):
    print('********Starting training with the following configuration :********')
    pp.pprint(args)

    m_backbone = load_backbone(args)
    model = ActionNet(backbone=m_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    n_params = sum([p.numel() for p in model.parameters()])
    print('Number of parameters: %d' % n_params)
    
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    jaad = JAADDataset(data_path=args.data_path)
    train_loader = DataLoader(jaad, **trainloader_params)

    optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()), "lr": args.lr_backbone},
                  {"params": filter(lambda p: p.requires_grad, model.head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone, 
                    weight_decay=args.weight_decay
        )
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    print('INFO: Training on {} batches'.format(len(train_loader)))

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs))
        model.train()
        for idx, (batch, label) in tqdm(enumerate(train_loader)):
            if torch.cuda.is_available():
                batch = batch.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if idx % opts.freq == 0:
                print('Epoch {}, batch {}, loss {}'.format(epoch, idx, loss.item()))
    
        scheduler.step()
    
    print('Finished training')


if __name__ == '__main__':
    opts = parse_args()
    args = get_config(opts.config)
    train(args,opts)