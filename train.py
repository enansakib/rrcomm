#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import shutil
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from torch.optim import lr_scheduler
from utils import video_transforms
import models
import datasets
from utils.AdamW import AdamW
from utils.helper import AverageMeter, accuracy

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Action Recognition')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--dataset', '-d', default='rrcomm',
                    choices=["rrcomm"],
                    help='dataset: rrcomm')
parser.add_argument('--arch', '-a', default='rrcommnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rrcommnet)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--iter-size', default=16, type=int,
                    metavar='I', help='iter size to reduce memory usage (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
parser.add_argument('--print-freq', default=400, type=int,
                    metavar='N', help='print frequency (default: 400)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 1)')
parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments in dataloader (default: 1)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-c', '--continue', dest='contine', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--input-skip', '--is', default=0, type=int,
                    choices=[0,1], help='skipping mechanism: 0 | 1 (default: 0)')

parser.add_argument('--categories', '--nc', default=15, type=int, help='number of action categories')


best_prec1 = 0
best_loss = 30
HALF = False
training_continue = False

def main():
    global args, best_prec1, model, writer, best_loss, length, width, height, input_size, scheduler, backbone, ckt_name
    args = parser.parse_args()
    training_continue = args.contine
    
    scale = 0.5
        
    print('scale: %.1f' %(scale))
    
    input_size = int(224 * scale)
    width = int(340 * scale)
    height = int(256 * scale)

    if args.dataset=='rrcomm':
        dataset='./datasets/rrcomm_frames'
    else:
        print("No convenient dataset entered, exiting....")
        return 0
    
    cudnn.benchmark = True

    if args.input_skip == 0:
        length=64
        channel_depth=4
    elif args.input_skip == 1:
        length=32
        channel_depth=2

    # Data transforming
    is_color = True
    scale_ratios = [1.0, 0.875, 0.75, 0.66]    
    clip_mean = [114.7748, 107.7354, 99.4750] * args.num_seg * length
    clip_std = [1, 1, 1] * args.num_seg * length        

    
    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)

    train_transform = video_transforms.Compose([
            video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor2(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            video_transforms.CenterCrop((input_size)),
            video_transforms.ToTensor2(),
            normalize,
        ])

    # data loading
    train_setting_file = "train_split%d.txt" % (args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    val_setting_file = "val_split%d.txt" % (args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    train_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                    source=train_split_file,
                                                    phase="train",
                                                    is_color=is_color,
                                                    new_length=length,
                                                    new_width=width,
                                                    new_height=height,
                                                    video_transform=train_transform,
                                                    num_segments=args.num_seg,
                                                    input_skip=args.input_skip)
    
    val_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                  source=val_split_file,
                                                  phase="val",
                                                  is_color=is_color,
                                                  new_length=length,
                                                  new_width=width,
                                                  new_height=height,
                                                  video_transform=val_transform,
                                                  num_segments=args.num_seg,
                                                  input_skip=args.input_skip)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    backbone = "ResNeXt101"
    ckt_name = args.dataset+"_"+args.arch+ str(args.input_skip) + "_" + backbone + "_split"+str(args.split)
    saveLocation="./checkpoints/"+ckt_name
    logLocation=saveLocation+'/logs'
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    if not os.path.exists(logLocation):
        os.makedirs(logLocation)

    writer = SummaryWriter(logLocation)
   
    # create model

    if args.evaluate:
        print("Building validation model ... ")
        model = build_model_validate(channel_depth)
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif training_continue:
        model, startEpoch, optimizer, best_prec1 = build_model_continue(channel_depth)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            #param_group['lr'] = lr
        print("Continuing with best precision: %.3f and start epoch %d and lr: %f" %(best_prec1,startEpoch,lr))
    else:
        print("Building model with ADAMW...")
        model = build_model(channel_depth)
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
        startEpoch = 0

    
    if HALF:
        model.half()  # convert to half precision
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    
    print("Model %s is loaded." % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, verbose=True)
    

    print("Saving everything to directory %s." % (saveLocation))
    

    if args.evaluate:
        prec1,prec3,lossClassification = validate(val_loader, model, criterion)
        return

    for epoch in range(startEpoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = 0.0
        lossClassification = 0
        if (epoch + 1) % args.save_freq == 0:
            prec1,prec3,lossClassification = validate(val_loader, model, criterion)
            writer.add_scalar('data/top1_validation', prec1, epoch)
            writer.add_scalar('data/top3_validation', prec3, epoch)
            writer.add_scalar('data/classification_loss_validation', lossClassification, epoch)
            scheduler.step(lossClassification)
        # remember best prec@1 and save checkpoint
        
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)      

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            if is_best:
                print("saving best checkpoint.")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_name, saveLocation)
    
    checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_name, saveLocation)
    writer.export_scalars_to_json("./"+ckt_name+"_all_scalars.json")
    writer.close()

def build_model(channel_depth):
    model_path = './weights/resnext-101.pth' 
        
    if args.dataset=='rrcomm':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=args.categories, length=args.num_seg, depth=channel_depth)
    
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model)
    model = model.cuda()
    
    return model

def build_model_validate(channel_depth):
    modelLocation="./checkpoints/"+ckt_name
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='rrcomm':
        model=models.__dict__[args.arch](modelPath='', num_classes=args.categories, length=args.num_seg, depth=channel_depth)

    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 

    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval() 
    return model

def build_model_continue(channel_depth):
    modelLocation="./checkpoints/"+ckt_name
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='rrcomm':
        model=models.__dict__[args.arch](modelPath='', num_classes=args.categories,length=args.num_seg, depth=channel_depth)
   
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 
        
    model.load_state_dict(params['state_dict'])
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])
    
    startEpoch = params['epoch']
    best_prec = params['best_prec1']
    return model, startEpoch, optimizer, best_prec


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch_classification = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter=0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
        
        if HALF:
            inputs = inputs.cuda().half()
        else:
            inputs = inputs.cuda()
        targets = targets.cuda()
        output, input_vectors, sequenceOut, maskSample = model(inputs)
        

        prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
        acc_mini_batch += prec1.item()
        acc_mini_batch_top3 += prec3.item()
        
        lossClassification = criterion(output, targets)
        
        lossClassification = lossClassification / args.iter_size
        
        totalLoss=lossClassification 
        loss_mini_batch_classification += lossClassification.data.item()
        totalLoss.backward()
        totalSamplePerIter +=  output.size(0)
        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
            top1.update(acc_mini_batch/args.iter_size, totalSamplePerIter)
            top3.update(acc_mini_batch_top3/args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch_classification = 0
            acc_mini_batch = 0
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter = 0.0
            
        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f' %(i,batch_time.avg,lossesClassification.avg))
          
    print('(TRAINING) Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
          .format(epoch = epoch, top1=top1, top3=top3, lossClassification=lossesClassification))
          
    writer.add_scalar('data/classification_loss_training', lossesClassification.avg, epoch)
    writer.add_scalar('data/top1_training', top1.avg, epoch)
    writer.add_scalar('data/top3_training', top3.avg, epoch)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
                
            if HALF:
                inputs = inputs.cuda().half()
            else:
                inputs = inputs.cuda()
            targets = targets.cuda()
    
            # compute output
            output, input_vectors, sequenceOut, _ = model(inputs)
            
            
            lossClassification = criterion(output, targets)
    
            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
            
            lossesClassification.update(lossClassification.data.item(), output.size(0))
            
            top1.update(prec1.item(), output.size(0))
            top3.update(prec3.item(), output.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    
        print('(VALIDATION) Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n' 
              .format(top1=top1, top3=top3, lossClassification=lossesClassification))

    return top1.avg, top3.avg, lossesClassification.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)



if __name__ == '__main__':
    main()