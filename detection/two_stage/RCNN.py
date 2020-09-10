import numpy as np
import tensorflow as tf
import utils as utils
import sys
sys.path.append('../')
import backbone_det as backbone
from config_det import cfg
from RPN import RPN
from ROI_pooling import ROI_pooling


class RCNN(object):
    def __init__(self, input_data, trainable):
        self.input_data = input_data
        self.trainable = trainable
        self.RPN_Network = RPN(self.input_data, self.trainable)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors  = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.pre_nms_top_n = cfg.RCNN.PRE_NMS_TOP_N
        self.post_nms_top_n = cfg.RCNN.POST_NMS_TOP_N
        self.pooling_size = cfg.RCNN.ROI_POOLING
        self.ROI_sbbox = ROI_pooling(self.RPN_Network.small_feature_map, self.pooling_size[2])
        self.ROI_mbbox = ROI_pooling(self.RPN_Network.medium_feature_map, self.pooling_size[1])
        self.ROI_lbbox = ROI_pooling(self.RPN_Network.large_feature_map, self.pooling_size[0])
        
        
        
    def compute_RPN_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        return self.RPN_Network.compute_loss(label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox)
    
    def RPN_result(self):
        return self.RPN_Network.pred_sbbox, self.RPN_Network.pred_mbbox, self.RPN_Network.pred_lbbox
    
    def compute_classi_loss(self, post_sbbox, post_mbbox, post_lbbox, label_sbbox, label_mbbox, label_lbbox, respond_sbbox, respond_mbbox, respond_lbbox):
        return self.ROI_sbbox.compute_loss(post_sbbox, label_sbbox, respond_sbbox) + self.ROI_mbbox.compute_loss(post_mbbox, label_mbbox, respond_mbbox) + self.ROI_lbbox.compute_loss(post_lbbox, label_lbbox, respond_lbbox)
        
        