import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer
from backbone import res34, res50, res101, res18

#strucrue of res34
class Resnet34(object):
    def __init__(self, input_data, trainable, classes, keep_prob):
        self.trainable = trainable
        self.input_data = input_data
        self.num_classes = classes
        self.keep_prob = keep_prob
        self.preds, self.last_layer = self.build_network()
        
        #print self.input_data.get_shape()

    def build_network(self):
        with tf.variable_scope('backbone'):
            _, _, _, input_ = res34.Res34(self.input_data, self.trainable)
        
        avg_pool = tf.reduce_mean(input_, (1, 2))

        prob = fc_layer(avg_pool, name='fc_layer', trainable=self.trainable, num_classes=self.num_classes, rate=self.keep_prob)
        
        #print prob.get_shape()
        
        return prob, input_


    def compute_loss(self, labels):
        loss_val = tf.reduce_mean(-tf.reduce_sum(tf.log(tf.clip_by_value(self.preds,1e-10,1.0)) * labels, reduction_indices=[1]))
        correct = tf.equal(tf.argmax(self.preds, 1), tf.argmax(labels, 1))
        accurate = tf.reduce_mean(tf.cast(correct, tf.float32))
        return loss_val, accurate
    
    def predict(self):
        return self.preds
    
    def cam(self):
        return self.last_layer
    


















