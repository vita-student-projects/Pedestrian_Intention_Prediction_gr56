import os
import numpy as np
import time
import sys
import argparse
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
from lib.data.dataset_jaad import JAADDataset
from lib.model.model_action import ActionNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/jaad/JAAD_train.yaml')
    parser.add_argument('-f', '--freq', type=int, default=100)
    parser.add_argument('-c','--checkpoint', action='store_true')
    parser.add_argument('-e','--eval', action='store_true')
    parser.add_argument('-cp', '--checkpoint_path', type=str,default='checkpoints')
    opts = parser.parse_args()
    return opts

def train(args,opts):
    print('INFO: Starting training with the following parameters')
    pp.pprint(args)

    print('INFO: Creating model')
    m_backbone = load_backbone(args)
    model = ActionNet(backbone=m_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    n_params = sum([p.numel() for p in model.parameters()])
    print('INFO: Number of parameters: %d' % n_params)
    
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
    }

    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 4,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'drop_last': False
    }

    print('INFO: Loading data')
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
    st = 0

    if opts.checkpoint:
        chk_filename = os.path.join(opts.checkpoint_path, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            print('INFO: Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

    print('INFO: Starting training ...')
    for epoch in range(st,args.epochs):
        print('INFO: Epoch {}/{}'.format(epoch, args.epochs))
        losses_train = AverageMeter()
        acc = AverageMeter()
        batch_time = AverageMeter()
        model.train() #put the model in training mode

        end = time.time()
        for idx, (batch, label) in tqdm(enumerate(train_loader)):
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
            acc.update(accuracy(output, label), batch_size)
            batch_time.update(time.time()-end)
            end = time.time()
            if (idx+1) % opts.freq == 0:
                print('', end='\r')
                print('INFO: Batch:[{0}/{1}] '
                      'Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss: {loss.val:.3f} ({loss.avg:.3}) '
                      'Accuracy: {acc.val:.3f} ({acc.avg:.3f})'.format(idx+1, len(train_loader), batch_time=batch_time, loss=losses_train, acc=acc),)
                sys.stdout.flush() #print directly
        
        print('INFO: Starting testing for Epoch {}/{}'.format(epoch, args.epochs))
        test_loss, acc = validate(test_loader, model, criterion)
        print('INFO: Testing done')
        print('Loss: {loss:.4f} Accuracy: {acc:.3f}'.format(loss=test_loss, acc=acc))
        scheduler.step()
        

        chk_path = os.path.join(opts.checkpoint_path, 'latest_epoch.bin')
        print('INFO: Saving checkpoint to', chk_path)
        torch.save({
            'epoch': epoch+1,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, chk_path)
    
    print('INFO: Finished training')

    #save the model
    print('INFO: Saving model')
    model_path = os.path.join(args.saved_model, 'jaad_model.ckpt')
    torch.save(model.state_dict(), model_path)

def validate(test_loader, model, criterion, freq=50):
    '''
    Function used to evaluate the model on the test set
    Input : test_loader, model, criterion
    Output : avg(losse), avg(acc)
    '''
    model.eval() #put the model in eval mode

    #metric we keep track off
    losses = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():
        for idx, (batch, label) in tqdm(enumerate(test_loader)):
            batch_size = len(batch)
            if torch.cuda.is_available():
                label = label.cuda()
                batch = batch.cuda()
            output = model(batch)
            loss = criterion(output, label)

            #update the metrics
            losses.update(loss.item(), batch_size)
            acc.update(accuracy(output, label), batch_size)
                
    return losses.avg, acc.avg

def evaluate(args,opts):
    print('INFO: Evaluating model')

    m_backbone = load_backbone(args)
    model = ActionNet(backbone=m_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 4,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    jaad_ts = JAADDataset(data_path=args.data_path,is_train=False)
    test_loader = DataLoader(jaad_ts, **testloader_params)
    model.load_state_dict(torch.load('trained_model/jaad_model.ckpt', map_location=lambda storage, loc: storage), strict=True)
    print('INFO: Evaluating on {} batches'.format(len(test_loader)))
    test_loss, acc = validate(test_loader, model, criterion)
    print('INFO: Evaluation done')
    print('Loss {loss:.4f} Acc@1 {acc:.3f}'.format(loss=test_loss, acc=acc))


if __name__ == '__main__':
    opts = parse_args()
    args = get_config(opts.config)
    if opts.eval:
        evaluate(args,opts)
    else:
        train(args,opts)