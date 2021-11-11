# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from utils.common_tools import call_parse_args
from models.config import model_from_file, model_from_list, model
from utils.roidb import combined_roidb
from utils.datasets import sampler, roibatchLoader

def check_set_dataset(args):
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_models = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_models = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_models = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_models = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_models = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

def load_set_model(args):
    args.model_file = "./models/{}_ls.yml".format(args.net) if args.large_scale else "./models/{}.yml".format(args.net)
    if args.model_file is not None:
      model_from_file(args.model_file)
    if args.set_models is not None:
      model_from_list(args.set_models)
    print('Using config:')
    pprint.pprint(model)
    np.random.seed(model.RNG_SEED)

def load_set_dataset(args):
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    model.TRAIN.USE_FLIPPED = True
    model.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                              sampler=sampler_batch, num_workers=args.num_workers)
    return imdb, dataloader

def init_train(args, imdb):
    # step1  initial data
     # step1.1 initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
     # step1.2 ship to cuda
    if args.cuda:
      im_data = im_data.cuda()
      im_info = im_info.cuda()
      num_boxes = num_boxes.cuda()
      gt_boxes = gt_boxes.cuda()
     # step1.3 make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    # step2 initial model
    if args.cuda:
      model.CUDA = True
     # step2.1 initilize the network here.
    if args.net == 'vgg16':
      fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()
    fasterRCNN.create_architecture()
    if args.cuda:
      fasterRCNN.cuda()
     # step2.2 initilize parameters.  
    lr = model.TRAIN.LEARNING_RATE
    lr = args.lr
     # tr_momentum = model.TRAIN.MOMENTUM
     #tr_momentum = args.momentum
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
      if value.requires_grad:
        if 'bias' in key:
          params += [{'params':[value],'lr':lr*(model.TRAIN.DOUBLE_BIAS + 1), \
                  'weight_decay': model.TRAIN.BIAS_DECAY and model.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': model.TRAIN.WEIGHT_DECAY}]

    # step3 initial optimizer
    if args.optimizer == "adam":
      lr = lr * 0.1
      optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
      optimizer = torch.optim.SGD(params, momentum=model.TRAIN.MOMENTUM)

    return  

def train(args):

    # step1 load datas
    imdb, dataloader = load_set_dataset(args)

    # step2 initial train   value and model
    init_train(args, imdb)








    if args.resume:
      load_name = os.path.join(output_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
      print("loading checkpoint %s" % (load_name))
      checkpoint = torch.load(load_name)
      args.session = checkpoint['session']
      args.start_epoch = checkpoint['epoch']
      fasterRCNN.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr = optimizer.param_groups[0]['lr']
      if 'pooling_mode' in checkpoint.keys():
        model.POOLING_MODE = checkpoint['pooling_mode']
      print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
      fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
      from tensorboardX import SummaryWriter
      logger = SummaryWriter("logs")

"""
    for epoch in range(args.start_epoch, args.max_epochs + 1):
      # setting to train mode
      fasterRCNN.train()
      loss_temp = 0
      start = time.time()

      if epoch % (args.lr_decay_step + 1) == 0:
          adjust_learning_rate(optimizer, args.lr_decay_gamma)
          lr *= args.lr_decay_gamma

      data_iter = iter(dataloader)
      for step in range(iters_per_epoch):
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
             + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()

        if step % args.disp_interval == 0:
          end = time.time()
          if step > 0:
            loss_temp /= (args.disp_interval + 1)

          if args.mGPUs:
            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_box = rpn_loss_box.mean().item()
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_box = RCNN_loss_bbox.mean().item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
          else:
            loss_rpn_cls = rpn_loss_cls.item()
            loss_rpn_box = rpn_loss_box.item()
            loss_rcnn_cls = RCNN_loss_cls.item()
            loss_rcnn_box = RCNN_loss_bbox.item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt

          print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                  % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
          if args.use_tfboard:
            info = {
              'loss': loss_temp,
              'loss_rpn_cls': loss_rpn_cls,
              'loss_rpn_box': loss_rpn_box,
              'loss_rcnn_cls': loss_rcnn_cls,
              'loss_rcnn_box': loss_rcnn_box
            }
            logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

          loss_temp = 0
          start = time.time()


      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': model.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))

    if args.use_tfboard:
      logger.close()
"""


if __name__ == '__main__':
    # step1 config parameters
    args = call_parse_args()
    print('Called with args:')
    print(args)

    # step2 set path of dataset, set the scale and ratio of anchor, and max number of groundtruth boxes 
    check_set_dataset(args)

    # step3 load model file and set model 
    load_set_model(args)

    # step4 torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
      print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # step5 train
    train(args)


