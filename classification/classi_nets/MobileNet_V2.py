import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer
from backbone import res34, res50, res101, res18, MobileNet_V1, MobileNet_V2
from loss import MSE, Cross_entropy

class Mobilenet_v2(object):
    def __init__(self, input_data, trainable, classes, keep_prob):
        self.trainable = trainable
        self.input_data = input_data
        self.num_classes = classes
        self.keep_prob = keep_prob
        self.preds, self.last_layer, self.cam_weights = self.build_network()
        
        #print self.input_data.get_shape()

    def build_network(self):
        with tf.variable_scope('backbone'):
            _, _, _, _, _, _, input_ = MobileNet_V2.MobileNet_V2(self.input_data, self.trainable)
        
        avg_pool = tf.reduce_mean(input_, (1, 2))

        prob, cam_weights = fc_layer(avg_pool, name='fc_layer', trainable=self.trainable, num_classes=self.num_classes, rate=self.keep_prob)
        
        #print prob.get_shape()
        
        return prob, input_, cam_weights


    def compute_loss(self, labels):
        loss_val = Cross_entropy(self.preds, labels)
        correct = tf.equal(tf.argmax(self.preds, 1), tf.argmax(labels, 1))
        accurate = tf.reduce_mean(tf.cast(correct, tf.float32))
        return loss_val, accurate
    
    def predict(self):
        return self.preds
    
    def cam(self):
        cam_bf_relu = tf.reduce_mean(self.last_layer * self.cam_weights[:, 1], axis=-1)
        cam_bf_relu = tf.nn.relu(cam_bf_relu)
        return tf.nn.relu(cam_bf_relu, name='cam')