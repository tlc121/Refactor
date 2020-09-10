import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer


#strucrue of res18
class Vgg(object):
    def __init__(self, input_data, trainable, classes, keep_prob):
        self.trainable = trainable
        self.input_data = input_data
        self.num_classes = classes
        self.keep_prob = keep_prob
        self.preds, self.last_layer, self.cam_weights = self.build_network()
        
        #print self.input_data.get_shape()

    def build_network(self):
        with tf.variable_scope('backbone'):
            input_ = convolutional(self.input_data, filter_shape=[3,3,1,8], name='conv1_1', bn=False, trainable=self.trainable)

            input_ = convolutional(input_, filter_shape=[3,3,8,16], name='conv1_2', downsample=True, bn=False, trainable=self.trainable)

            input_ = convolutional(input_, filter_shape=[3,3,16,32], name='conv2_1', bn=False, trainable=self.trainable)

            input_ = convolutional(input_, filter_shape=[3,3,32,32], name='conv2_2', downsample=True, bn=False, trainable=self.trainable)

            nput_ = convolutional(input_, filter_shape=[3,3,32,32], name='conv3_1', bn=False, trainable=self.trainable)

            input_ = convolutional(input_, filter_shape=[3,3,32,32], name='conv3_2', downsample=True, bn=False, trainable=self.trainable)

            nput_ = convolutional(input_, filter_shape=[3,3,32,32], name='conv4_1', bn=False, trainable=self.trainable)

            input_ = convolutional(input_, filter_shape=[3,3,32,64], name='conv4_2', bn=False, trainable=self.trainable)

        
        avg_pool = tf.reduce_mean(input_, (1, 2))

        prob, cam_weights = fc_layer(avg_pool, name='fc_layer', trainable=self.trainable, num_classes=self.num_classes, rate=self.keep_prob)
        
        #print prob.get_shape()
        
        return prob, input_, cam_weights


    def compute_loss(self, labels):
        loss_val = tf.reduce_mean(-tf.reduce_sum(tf.log(tf.clip_by_value(self.preds,1e-10,1.0)) * labels, reduction_indices=[1]))
        correct = tf.equal(tf.argmax(self.preds, 1), tf.argmax(labels, 1))
        accurate = tf.reduce_mean(tf.cast(correct, tf.float32))
        return loss_val, accurate
    
    def predict(self):
        return self.preds
    
    def cam(self):
        return tf.reduce_sum(self.last_layer * self.cam_weights[:, 1], axis=-1)
    


















