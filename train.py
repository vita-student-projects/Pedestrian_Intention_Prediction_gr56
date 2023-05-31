import os
import shutil
import time
import sys
import errno
import argparse
from tqdm import tqdm
import pprint as pp # pretty print
import tensorboardX
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.dataset_jaad import KPJAADDataset
from lib.model.model_action import ActionNet
    

def parse_args():
    '''
    Function used to parse the launch arguments
    Input : None
    Output : opts (launch arguments)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/JAAD_train.yaml', help='Path to the config file.')
    parser.add_argument('-f', '--freq', type=int, default=100, help='Frequency of printing training metrics.')
    parser.add_argument('-c','--checkpoint', action='store_true', help='Continue training from a checkpoint.')
    parser.add_argument('-e','--eval', action='store_true', help='Evaluate the model.')
    opts = parser.parse_args()
    return opts

def train(args,opts):
    '''
    Function used to train the model
    Input : args (config arguments)
            opts (launch arguments)
    Output : None
    '''
    print('INFO: Starting training with the following parameters')
    pp.pprint(args)

    #create the checkpoint directory if it does not exist
    try:
        os.makedirs('checkpoints')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory: checkpoints')

    #clear the logs directory if we are not resuming from a checkpoint
    if not opts.checkpoint:
        try:
            shutil.rmtree(args.logs_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise RuntimeError("Unable to delete {0} because of {1}.".format(e.filename, e.strerror))
    
    #create the logs directory if it does not exist
    try:
        os.makedirs(args.logs_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create logs directory: {}'.format(args.logs_path))

    #initialize a logger
    train_writer = tensorboardX.SummaryWriter(args.logs_path)

    print('INFO: Creating model')
    m_backbone = load_backbone(args)
    model = ActionNet(backbone=m_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = nn.CrossEntropyLoss()

    #move the model to the GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    n_params = sum([p.numel() for p in model.parameters()])
    print('INFO: Number of parameters: %d' % n_params)
    
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True,
        'drop_last': False
    }

    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 2,
          'pin_memory': True,
          'persistent_workers': True,
          'drop_last': False
    }

    print('INFO: Loading data')
    jaad_tr = KPJAADDataset(data_path=args.data_path,is_train=True)
    jaad_ts = KPJAADDataset(data_path=args.data_path,is_train=False)
    train_loader = DataLoader(jaad_tr, **trainloader_params)
    test_loader = DataLoader(jaad_ts, **testloader_params)

    #create the optimizer, we use AdamW for both the backbone and the head
    optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()), "lr": args.lr_backbone},
                  {"params": filter(lambda p: p.requires_grad, model.head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone, 
                    weight_decay=args.weight_decay
        )
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    print('INFO: Training on {} batches'.format(len(train_loader)))

    #load the checkpoint if we are resuming training
    st = 0
    if opts.checkpoint:
        if os.path.exists(args.chk_path):
            print('INFO: Loading checkpoint', args.chk_path)
            checkpoint = torch.load(args.chk_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

    best_acc = 0
    print('INFO: Starting training ...')
    for epoch in range(st,args.epochs):
        print('INFO: Epoch {}/{}'.format(epoch+1, args.epochs))

        #metric we keep track off
        losses_train = AverageMeter()
        acc = AverageMeter()
        batch_time = AverageMeter()
        f1 = AverageMeter()

        #put the model in training mode
        model.train() 

        end = time.time()
        for idx, (batch, label) in tqdm(enumerate(train_loader)):
            batch_size = len(batch)

            #move the batch to the GPU if available
            if torch.cuda.is_available():
                batch = batch.cuda()
                label = label.cuda()

            #forward pass
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, label)

            #backward pass
            loss.backward()
            optimizer.step()
            
            #update the metrics
            losses_train.update(loss.item(), batch_size)
            acc.update(accuracy(output, label), batch_size)
            batch_time.update(time.time()-end)
            f1.update(f1_score(label.cpu().numpy(), output.argmax(dim=1).cpu().numpy(), average='macro'), batch_size) #macro mode computes the metric independently for 
                                                                                                                      #each class and then takes the average 
                                                                                                                      #(hence treating all classes equally, even if some are unbalanced)
            end = time.time()

            #print the metrics every opts.freq batches
            if (idx+1) % opts.freq == 0:
                print('', end='\r') #clear the line
                print('INFO: Batch:[{0}/{1}] '
                      'Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss: {loss.val:.3f} ({loss.avg:.3}) '
                      'Accuracy: {acc.val:.3f} ({acc.avg:.3f})'
                      'F1: {f1.val:.3f} ({f1.avg:.3f})'.format(idx+1, len(train_loader), batch_time=batch_time, loss=losses_train, acc=acc, f1=f1))
                      
                sys.stdout.flush() #print directly
        
        print('INFO: Starting testing for Epoch {}/{}'.format(epoch+1, args.epochs))
        test_loss, test_acc, test_f1 = validate(test_loader, model, criterion) #evaluate the model on the test set
        print('INFO: Testing done')
        print('Loss: {loss:.4f} Accuracy: {acc:.3f}'.format(loss=test_loss, acc=test_acc))
        scheduler.step()
        
        #write the metrics to tensorboard for visualization
        train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
        train_writer.add_scalar('train_acc', acc.avg, epoch + 1)
        train_writer.add_scalar('train_f1', f1.avg, epoch + 1)
        train_writer.add_scalar('test_loss', test_loss, epoch + 1)
        train_writer.add_scalar('test_acc', test_acc, epoch + 1)
        train_writer.add_scalar('test_f1', test_f1, epoch + 1)
        
        #saving the model checkpoint
        print('INFO: Saving checkpoint to', args.chk_path)
        torch.save({
            'epoch': epoch+1,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, args.chk_path)

        #saving the best model checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch+1,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }, args.best_chk_path)

    print('INFO: Finished training')

    #evaluate the model structure if we are not resuming from a checkpoint or using a GPU
    if not opts.checkpoint and not torch.cuda.is_available():
        print('INFO: Evaluating model structure')
        batch = next(iter(train_loader))[0]
        train_writer.add_graph(model, batch)
        print('INFO: Evaluation done')

def validate(test_loader, model, criterion):
    '''
    Function used to evaluate the model on the test set
    Input : test_loader, model, criterion
    Output : avg(losse), avg(acc)
    '''
    model.eval() #put the model in eval mode

    #metric we keep track off
    losses = AverageMeter()
    accu = AverageMeter()
    f1 = AverageMeter()

    #disable gradient computation
    with torch.no_grad():
        for idx, (batch, label) in tqdm(enumerate(test_loader)):
            batch_size = len(batch)

            #move the batch to the GPU if available
            if torch.cuda.is_available():
                label = label.cuda()
                batch = batch.cuda()

            #forward pass
            output = model(batch)
            loss = criterion(output, label)

            #update the metrics
            losses.update(loss.item(), batch_size)
            accu.update(accuracy(output, label), batch_size)
            f1.update(f1_score(label.cpu().numpy(), output.argmax(dim=1).cpu().numpy(), average='macro'), batch_size) 

    return losses.avg, accu.avg, f1.avg

def evaluate(args):
    '''
    Function used to evaluate the model on the test set from a checkpoint
    Input : args (config arguments)
    Output : None
    '''
    print('INFO: Evaluating model')
    print('INFO: Loading model')
    m_backbone = load_backbone(args)
    model = ActionNet(backbone=m_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = nn.CrossEntropyLoss()

    #move the model to the GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 2,
          'pin_memory': True,
          'persistent_workers': True,
          'drop_last': False
    }

    jaad_ts = KPJAADDataset(data_path=args.data_path,is_train=False)
    test_loader = DataLoader(jaad_ts, **testloader_params)

    #load the checkpoint
    model.load_state_dict(torch.load(args.chk_path, map_location=lambda storage, loc: storage)['model'], strict=True)

    print('INFO: Evaluating on {} batches'.format(len(test_loader)))
    test_loss, acc = validate(test_loader, model, criterion) #evaluate the model on the test set
    print('INFO: Evaluation done')
    print('INFO: Loss: {loss:.4f} Acc: {acc:.3f}'.format(loss=test_loss, acc=acc))


if __name__ == '__main__':
    opts = parse_args() #parse the launch arguments
    args = get_config(opts.config) #parse the config file

    #check if we are evaluating or training
    if opts.eval:
        evaluate(args)
    else:
        train(args,opts)