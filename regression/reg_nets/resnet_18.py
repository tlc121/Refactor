import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, fc_layer_reg
from backbone import res34, res50, res101, res18
from loss import MSE, Cross_entropy

#strucrue of res18
class Resnet18(object):
    def __init__(self, input_data, trainable, classes, keep_prob, scale):
        self.trainable = trainable
        self.input_data = input_data
        self.num_classes = classes
        self.keep_prob = keep_prob
        self.scale = scale
        self.preds, self.last_layer = self.build_network()
        
        #print self.input_data.get_shape()

    def build_network(self):
        with tf.variable_scope('backbone'):
            _, _, _, _, _, _, input_ = res18.Res18(self.input_data, self.trainable)
            
        avg_pool = tf.reduce_mean(input_, (1, 2))
        
       
        prob = fc_layer_reg(avg_pool, name='fc_layer', trainable=self.trainable, num_classes=self.num_classes, rate=self.keep_prob, scale=self.scale)
        
        #print prob.get_shape()
        
        return prob, input_


    def compute_loss(self, labels):
        loss_val = MSE(self.preds, labels)
        return loss_val
    
    def predict(self):
        return self.preds
    
    def cam(self):
        return self.last_layer
    


















