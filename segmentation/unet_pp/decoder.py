import tensorflow as tf
import sys
sys.path.append('../')
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, deconvolutional
from config_seg import cfg

def decoder(route_0, route_1, route_2, route_3, route_4, trainable):
    with tf.variable_scope('decoder'):
        num_filters = [32, 64, 128, 256, 512]
        out_channel = cfg.TRAIN.NUMCLASS
        
        #first part
        up_1_2 = deconvolutional(route_1, num_filters[0], 2, (2,2), name='deconv1')
        conv_1_2 = tf.concat([route_0, up_1_2], axis=-1)
        conv_1_2 = convolutional(conv_1_2, filter_shape=[3,3, conv_1_2.get_shape()[-1], num_filters[0]], name='conv1', trainable=trainable)
        
        
        
        #second part
        up_2_2 = deconvolutional(route_2, num_filters[1], 2, (2,2), name='deconv2_0')
        conv_2_2 = tf.concat([route_1, up_2_2], axis=-1)
        conv_2_2 = convolutional(conv_2_2, filter_shape=[3,3, conv_2_2.get_shape()[-1], num_filters[1]], name='conv2_0', trainable=trainable)
        
        up_1_3 = deconvolutional(conv_2_2, num_filters[0], 2, (2,2), name='deconv2_1')
        conv_1_3 = tf.concat([up_1_3, route_0, conv_1_2], axis=-1)
        conv_1_3 = convolutional(conv_1_3, filter_shape=[3,3, conv_1_3.get_shape()[-1], num_filters[0]], name='conv2_1', trainable=trainable)
        
        
        
        #third part
        up_3_2 = deconvolutional(route_3, num_filters[2], 2, (2,2), name='deconv3_0')
        conv_3_2 = tf.concat([up_3_2, route_2], axis=-1)
        conv_3_2 = convolutional(conv_3_2, filter_shape=[3,3, conv_3_2.get_shape()[-1], num_filters[2]], name='conv3_0', trainable=trainable)
        
        up_2_3 = deconvolutional(conv_3_2, num_filters[1], 2, (2,2), name='deconv3_1')
        conv_2_3 = tf.concat([up_2_3, route_1, conv_2_2], axis=-1)
        conv_2_3 = convolutional(conv_2_3, filter_shape=[3,3, conv_2_3.get_shape()[-1], num_filters[1]], name='conv3_1', trainable=trainable)
        
        up_1_4 = deconvolutional(conv_2_3, num_filters[0], 2, (2,2), name='deconv3_2')
        conv_1_4 = tf.concat([up_1_4, route_0, conv_1_2, conv_1_3], axis=-1)
        conv_1_4 = convolutional(conv_1_4, filter_shape=[3,3, conv_1_4.get_shape()[-1], num_filters[0]], name='conv3_3', trainable=trainable)
        
       
    
    
        #final part
        up_4_2 = deconvolutional(route_4, num_filters[3], 2, (2,2), name='deconv4_0')
        conv_4_2 = tf.concat([route_3, up_4_2], axis=-1)
        conv_4_2 = convolutional(conv_4_2, filter_shape=[3,3, conv_4_2.get_shape()[-1], num_filters[3]], name='conv4_0', trainable=trainable)
        
        up_3_3 = deconvolutional(conv_4_2, num_filters[2], 2, (2,2), name='deconv4_1')
        conv_3_3 = tf.concat([up_3_3, route_2, conv_3_2], axis=-1)
        conv_3_3 = convolutional(conv_3_3, filter_shape=[3,3, conv_3_3.get_shape()[-1], num_filters[2]], name='conv4_1', trainable=trainable)
        
        up_2_4 = deconvolutional(conv_3_3, num_filters[1], 2, (2,2), name='deconv4_2')
        conv_2_4 = tf.concat([up_2_4, route_1, conv_2_2, conv_2_3], axis=-1)
        conv_2_4 = convolutional(conv_2_4, filter_shape=[3,3, conv_2_4.get_shape()[-1], num_filters[1]], name='conv4_2', trainable=trainable)
        
        up_1_5 = deconvolutional(conv_2_4, num_filters[0], 2, (2,2), name='deconv4_3')
        conv_1_5 = tf.concat([up_1_5, route_0, conv_1_2, conv_1_3, conv_1_4], axis=-1)
        conv_1_5 = convolutional(conv_1_5, filter_shape=[3,3, conv_1_5.get_shape()[-1], num_filters[0]], name='conv4_3', trainable=trainable)
        
        print conv_1_5.get_shape()[-1]
        conv_5 = convolutional(conv_1_5, filter_shape=[3,3, conv_1_5.get_shape()[-1], out_channel], activation=False, bn=False, name='conv5', trainable=trainable)
        
        output = tf.nn.sigmoid(conv_5, name='op_to_store')
        print output.get_shape().as_list()
        return output