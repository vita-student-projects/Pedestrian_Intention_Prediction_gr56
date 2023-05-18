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
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 4,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    jaad_tr = JAADDataset(data_path=args.data_path,is_train=True)
    jaad_ts = JAADDataset(data_path=args.data_path,is_train=False)
    train_loader = DataLoader(jaad_tr, **trainloader_params)
    test_loader = DataLoader(jaad_ts, **testloader_params)

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
        losses_train = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model.train() #put the model in training mode

        end = time.time()
        for idx, (batch, label) in tqdm(enumerate(train_loader)):
            data_time.update(time.time()-end)
            batch_size = len(batch)
            if torch.cuda.is_available():
                batch = batch.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            losses_train.update(loss.item(), batch_size)
            acc1 = accuracy(output, label, topk=(1,))
            top1.update(acc1[0], batch_size)
            batch_time.update(time.time()-end)
            batch_time.update(time.time()-end)
            end = time.time()
            if (idx+1) % opts.freq == 0:
                print('Train: [epoch: {0}][batch: {1}/{2}]\t'
                      'BT {batch_time.val} ({batch_time.avg})\t'
                      'DT {data_time.val} ({data_time.avg})\t'
                      'loss {loss.val} ({loss.avg})\t'
                      'Acc@1 {top1.val} ({top1.avg})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_train, top1=top1))
                sys.stdout.flush() #print directly

                #validate(test_loader, model, criterion,freq=10)

        scheduler.step()

        chk_path = os.path.join(args.checkpoint, 'latest_epoch.bin')
        print('Saving checkpoint to', chk_path)
        torch.save({
            'epoch': epoch+1,
            'lr': scheduler.get_last_lr(),
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, chk_path)
    
    print('Finished training')

    #save the model
    model_path = os.path.join(args.checkpoint, 'jaad_model.ckpt')
    torch.save(model.state_dict(), model_path)

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

    with torch.no_grad():
        end = time.time()
        for idx, (batch, batch) in tqdm(enumerate(test_loader)):
            batch_size = len(batch)
            if torch.cuda.is_available():
                label = label.cuda()
                batch = batch.cuda()
            output = model(batch)
            loss = criterion(output, label)
        
            #update the metrics
            losses.update(loss.item(), batch_size)
            acc1 = accuracy(output, label, topk=(1,))
            top1.update(acc1[0], batch_size)
            

            #measure the time since the time for the begining of the training on the batch
            batch_time.update(time.time()-end)
            end = time.time()

            if (idx+1) % freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val} ({batch_time.avg})\t'
                      'Loss {loss.val} ({loss.avg})\t'
                      'Acc@1 {top1.val} ({top1.avg})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
                
    return losses.avg, top1.avg


if __name__ == '__main__':
    opts = parse_args()
    args = get_config(opts.config)
    train(args,opts)