import tensorflow as tf
import sys
sys.path.append('../')
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, deconvolutional
from config_seg import cfg



def decoder(output, trainable):
    with tf.variable_scope('decoder'):
        out_channel = cfg.TRAIN.NUMCLASS
        
        conv_1 = convolutional(output, filter_shape=[3,3, output.get_shape()[-1], 256], name='conv1_1', trainable=trainable)
        conv_2 = convolutional(conv_1, filter_shape=[3,3, conv_1.get_shape()[-1], 256], name='conv1_2', trainable=trainable)
        conv_3 = convolutional(conv_2, filter_shape=[1,1, conv_2.get_shape()[-1], 256], name='conv1_3', trainable=trainable)
        
        dusample = convolutional(conv_3, filter_shape=[1,1, conv_3.get_shape()[-1], 256*16*16], name='conv1_4', trainable=trainable)
        shape_du = dusample.get_shape().as_list()
        b, h, w, c = shape_du[0], shape_du[1], shape_du[2], shape_du[3]
        #dusample = tf.reshape(dusample, (b, h*w*c))
        out = tf.reshape(dusample, (-1, h*16, w*16, c/256))
        conv_5 = convolutional(out, filter_shape=[3,3, out.get_shape()[-1], out_channel], activation=False, bn=False, name='conv5', trainable=trainable)
        output = tf.nn.sigmoid(conv_5, name='op_to_store')
        return output