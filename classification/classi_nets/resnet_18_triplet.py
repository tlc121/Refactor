import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, fc_layer_triplet
from backbone import res34, res50, res101, res18
from loss import MSE, Cross_entropy

#strucrue of res18
class Resnet18(object):
    def __init__(self, input_anchor, input_pos, input_neg, trainable, classes, keep_prob):
        self.trainable = trainable
        self.input_anchor = input_anchor
        self.input_pos = input_pos
        self.input_neg = input_neg
        self.num_classes = classes
        self.keep_prob = keep_prob
        self.prob_anchor, self.prob_pos, self.prob_neg = self.build_network()

    def build_network(self):
        with tf.variable_scope('backbone', reuse=tf.AUTO_REUSE):
            _, _, _, _, _, _, input_anchor = res18.Res18(self.input_anchor, self.trainable)
            _, _, _, _, _, _, input_pos = res18.Res18(self.input_pos, self.trainable)
            _, _, _, _, _, _, input_neg = res18.Res18(self.input_neg, self.trainable)
            
        avg_pool_anchor = tf.reduce_mean(input_anchor, (1, 2))
        avg_pool_pos = tf.reduce_mean(input_pos, (1, 2))
        avg_pool_neg = tf.reduce_mean(input_neg, (1, 2))
        
        prob_anchor = fc_layer_triplet(avg_pool_anchor, name='fc_layer', trainable=self.trainable, num_classes=128, rate=self.keep_prob)
        prob_pos = fc_layer_triplet(avg_pool_pos, name='fc_layer', trainable=self.trainable, num_classes=128, rate=self.keep_prob)
        prob_neg = fc_layer_triplet(avg_pool_neg, name='fc_layer', trainable=self.trainable, num_classes=128, rate=self.keep_prob)

        return prob_anchor, prob_pos, prob_neg
    
   
    def compute_loss(self, alpha=1.0):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(self.prob_anchor, self.prob_pos)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(self.prob_anchor, self.prob_neg)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        return loss
    
        
        
        

        
    


















