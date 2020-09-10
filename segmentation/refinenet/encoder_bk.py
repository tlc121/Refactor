import tensorflow as tf
import sys
sys.path.append('../')
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, deconvolutional
from config_seg import cfg



def encoder(input_data, trainable):
    with tf.variable_scope('backbone'):
        input_ = convolutional(input_data, filter_shape=[3,3,3,32], name='conv0_0', trainable=trainable)
        input_ = convolutional(input_, filter_shape=[3,3,32,32], name='conv0_1', trainable=trainable)
        route_0 = input_
        
        input_ = Max_Pooing(input_, name='max_pooling_1')
        input_ = convolutional(input_, filter_shape=[3,3,32,64], name='conv1_0', trainable=trainable)
        convolutional(input_, filter_shape=[3,3,64,64], name='conv1_1', trainable=trainable)
        route_1 = input_
        
        input_ = Max_Pooing(input_, name='max_pooling_2')
        input_ = convolutional(input_, filter_shape=[3,3,64,128], name='conv2_0', trainable=trainable)
        input_ = convolutional(input_, filter_shape=[3,3,128,128], name='conv2_1', trainable=trainable)
        route_2 = input_
        
        input_ = Max_Pooing(input_, name='max_pooling_3')
        input_ = convolutional(input_, filter_shape=[3,3,128,256], name='conv3_0', trainable=trainable)
        input_ = convolutional(input_, filter_shape=[3,3,256,256], name='conv3_1', trainable=trainable)
        route_3 = input_
        
        input_ = Max_Pooing(input_, name='max_pooling_4')
        input_ = convolutional(input_, filter_shape=[3,3,256,512], name='conv4_0', trainable=trainable)
        input_ = convolutional(input_, filter_shape=[3,3,512,512], name='conv4_1', trainable=trainable)
        route_4 = input_
        
        return route_0, route_1, route_2, route_3, route_4