#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict
import sys


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "/home/tanglc/1Refactor/detection/data/classes/ANN09.names"
__C.YOLO.ANCHORS                = "/home/tanglc/1Refactor/detection/data/anchors/tct_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "deconv"
__C.YOLO.ORIGINAL_WEIGHT        = "./tensorflow-yolov3/checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./tensorflow-yolov3/checkpoint/yolov3_coco_demo.ckpt"


__C.RCNN                       = edict()

__C.RCNN.PRE_NMS_TOP_N         = [412, 1650, 6600] #two-stage prediction bboxes selectoin
__C.RCNN.POST_NMS_TOP_N        = [62, 250, 1000] #two-stage prediction bboxses selection
__C.RCNN.IOU_THRESHOLD         = 0.7
__C.RCNN.ROI_POOLING          = [2, 4, 8]


# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = '/home/yuyue/yuyue/yolo_TCT/tensorflow-yolov3-multilabel/cut_6_patch_yolo/train_0817_三分类.txt'
__C.TRAIN.BATCH_SIZE            = 32
__C.TRAIN.INPUT_SIZE            = [224]
__C.TRAIN.RESPECT_RATIO            = 1.6
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.PRETRAIN_MODE         = 'backbone' #'whole' or 'backbone'
__C.TRAIN.BACKBONE            = 'darknet53'
__C.TRAIN.BACKBONE_PRETRAIN      = "/ssd2/tlc/pretrain_model/darknet53/darknet53_test_loss=6.9632-1"
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 1
__C.TRAIN.SAVE_PATH_PB          = '/hdd/sd5/tlc/TCT/Model_pb/focal_loss/'
__C.TRAIN.SAVE_PATH_CKPT         = '/hdd/sd5/tlc/TCT/Model_ckpt/focal_loss/'
__C.TRAIN.FISRT_STAGE_EPOCHS    = 10
__C.TRAIN.SECOND_STAGE_EPOCHS   = 15
__C.TRAIN.INITIAL_WEIGHT        = "/hdd/sd5/tlc/TCT/Model_ckpt/focal_loss/darknet53_test_loss=6.8387-1"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "/home/yuyue/yuyue/yolo_TCT/tensorflow-yolov3-multilabel/cut_6_patch_yolo/val_0817_三分类.txt"
__C.TEST.BATCH_SIZE             = 32
__C.TEST.INPUT_SIZE             = 224
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH         = "/home/tanglc/tensorflow-yolov3/data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL    = False
__C.TEST.WEIGHT_FILE            = "/hdd/sd5/tlc/TCT/Model_ckpt/focal_loss/darknet53_test_loss=7.9698-3"
__C.TEST.SHOW_LABEL             = False
__C.TEST.SCORE_THRESHOLD         = 0.3
__C.TEST.IOU_THRESHOLD          = 0.3






