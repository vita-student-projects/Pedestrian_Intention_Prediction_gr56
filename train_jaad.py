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

def validate(test_loader, model, criterion, freq):
    '''
    Function used to evaluate the model on the test set
    Input : test_loader, model, criterion
    Output : avg(losse), top1, top5
    '''
    model.eval() #put the model in eval mode

    #metric we keep track off
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)
            loss = criterion(output, batch_gt)
        
            #update the metrics
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1,5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            #measure the time since the time for the begining of the training on the batch
            batch_time.update(time.time()-end)
            end = time.time()

            if (idx+1) % freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
                
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    opts = parse_args()
    args = get_config(opts.config)
    train(args,opts)