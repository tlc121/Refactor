import numpy as np
import tensorflow as tf
import utils as utils
import core.common as common
import random
import backbone_det as backbone
from config_det import cfg
import sys
sys.path.append('../../')
from nms import nms
from core.common import fc_layer




class ROI_pooling(object):
    def __init__(self, conv_bbox, pooling_size):
        self.input_size = random.choice(cfg.TRAIN.INPUT_SIZE)
        self.conv_bbox = conv_bbox
        self.pooling_size = pooling_size
        
    
    def roi_pool(self, featureMaps, rois, im_dims):
        '''
        Regions of Interest (ROIs) from the Region Proposal Network (RPN) are
        formatted as:    (image_id, x1, y1, x2, y2)
        Note: Since mini-batches are sampled from a single image, image_id = 0s
        '''
        with tf.variable_scope('roi_pool'):
            box_ind = 0
            boxes = rois[:, 0:]

            normalization = tf.cast(tf.stack([im_dims[:, 1], im_dims[:, 0], im_dims[:, 1], im_dims[:, 0]], axis=1),
                                    dtype=tf.float32)  
            boxes = tf.div(boxes,normalization) 
 
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)

            crop_size = tf.constant([self.pooling_size*2, self.pooling_size*2])

            # ROI pool
            pooledFeatures = tf.image.crop_and_resize(image=featureMaps, boxes=boxes, box_ind=box_ind, crop_size=crop_size)
            pooledFeatures = tf.nn.max_pool(pooledFeatures, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #print('pooledFeatures', pooledFeatures)
        return pooledFeatures


    
    def classification_loss(self, bboxes_label, bboxes_predict, conv_bbox, respond_bbox):
        num_candidates = bboxes_predict.get_shape()[0]
        num_all_boxes = bboxes_label.get_shape()[0]
        num_classes = bboxes_label.get_shape()[-1] - 5
        loss = tf.constant(0.0)
        count = 0
        for idx in range(num_candidates):
            bbox_pred = bboxes_predict[idx]
            for idx_gt in range(num_all_boxes):
                bbox_label = bboxes_label[idx_gt]
                label_onehot = bboxes_label[5:]
                input_size = self.input_size
                feature_map_size = conv_bbox.get_shape()[1]
                ratio = tf.cast(tf.div(feature_map_size, input_size), dtype=tf.float32)
                conv_xywh = bbox_pred[:4] * ratio
                x, y, w, h = bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3]
#                 if w < self.pooling_size or h < self.pooling_size:
#                     continue
                feature_map = self.conv_bbox
                roi_feature_maps = self.roi_pool(feature_map, conv_xywh[tf.newaxis, ...], im_dims=np.array([[self.input_size, self.input_size]]))
                avg_pool = tf.reduce_mean(roi_feature_map, (1, 2))
                label_onehot = label_onehot[tf.newaxis, ...]
                prob = fc_layer(avg_pool, name='fc_layer', trainable=self.trainable, num_classes=num_classes, rate=1.0)
                prob_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = label_onehot, logits = prob)
                prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1])) * respond_bbox[idx, 0]
                loss += prob_loss
                #print tf.shape(loss)
                count += 1
                    
        return loss/count
                     
    def compute_loss(self, pred_bbox, label_bbox, respond_bbox):
        batch, num_bboxes, dim = pred_bbox.get_shape()[0], pred_bbox.get_shape()[1], pred_bbox.get_shape()[2]
        print label_bbox.get_shape().as_list()
        #h, w, c, dim = tf.shape(label_bbox)[1], tf.shape(label_bbox)[2], tf.shape(label_bbox)[3], tf.shape(label_bbox)[4]
        loss = tf.constant(0)
        for b in range(batch):
            #sorted_bbox = tf.gather(self.final_bbox[b,:,:], tf.argsort(self.final_bbox[b,:,4],direction = 'DESCENDING'))
            bboxes_label = tf.reshape(label_bbox[b, :, :, :, :], (num_bboxes, dim))
            loss += self.classification_loss(bboxes_label, pred_bbox, self.conv_bbox[b, :, :, :], respond_bbox[b, :, :])
            
        return loss/b   
            
            
        
        
        
        
        
        