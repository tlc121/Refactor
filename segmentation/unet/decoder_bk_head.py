import tensorflow as tf
import sys
sys.path.append('../')
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, deconvolutional
from config_seg import cfg

def decoder(route_0, route_1, route_2, route_3, route_4, input_, trainable):
    with tf.variable_scope('decoder'):
        out_channel = cfg.TRAIN.NUMCLASS
        
        avg_pool = tf.reduce_mean(input_, (1,2))
        prob, cam_weights = fc_layer(avg_pool, name='fc_layer', trainable=trainable, num_classes=2, rate=1.0)
        
        
        up_1= deconvolutional(route_4, route_4.get_shape()[-1]/2, 2, (2,2), name='deconv1')
        merge_1 = tf.concat([route_3, up_1], axis=-1)
        conv_1 = convolutional(merge_1, filter_shape=[3,3, merge_1.get_shape()[-1], merge_1.get_shape()[-1]/2], name='conv1_0', trainable=trainable)
        conv_1 = convolutional(conv_1, filter_shape=[3,3, conv_1.get_shape()[-1], conv_1.get_shape()[-1]], name='conv1_1', trainable=trainable)
    
        up_2 = deconvolutional(conv_1, conv_1.get_shape()[-1]/2, 2, (2,2), name='deconv2')
        merge_2 = tf.concat([route_2, up_2], axis=-1)
        conv_2 = convolutional(merge_2, filter_shape=[3,3, merge_2.get_shape()[-1], merge_2.get_shape()[-1]/2], name='conv2_0', trainable=trainable)
        conv_2 = convolutional(conv_2, filter_shape=[3,3, conv_2.get_shape()[-1], conv_2.get_shape()[-1]], name='conv2_1', trainable=trainable)
        
        up_3 = deconvolutional(conv_2, conv_2.get_shape()[-1]/2, 2, (2,2), name='deconv3')
        merge_3 = tf.concat([route_1, up_3], axis=-1)
        conv_3 = convolutional(merge_3, filter_shape=[3,3, merge_3.get_shape()[-1], merge_3.get_shape()[-1]/2], name='conv3_0', trainable=trainable)
        conv_3 = convolutional(conv_3, filter_shape=[3,3, conv_3.get_shape()[-1], conv_3.get_shape()[-1]], name='conv3_1', trainable=trainable)
        
        up_4 = deconvolutional(conv_3,conv_3.get_shape()[-1]/2, 2, (2,2), name='deconv4')
        merge_4 = tf.concat([route_0, up_4], axis=-1)
        conv_4 = convolutional(merge_4, filter_shape=[3,3, merge_4.get_shape()[-1], merge_4.get_shape()[-1]/2], name='conv4_0', trainable=trainable)
        conv_4 = convolutional(conv_4, filter_shape=[3,3, conv_4.get_shape()[-1], conv_4.get_shape()[-1]], name='conv4_1', trainable=trainable)
        
        conv_5 = convolutional(conv_4, filter_shape=[3,3, conv_4.get_shape()[-1], out_channel], activation=False, bn=False, name='conv5', trainable=trainable)
        
        output = tf.nn.sigmoid(conv_5, name='op_to_store')
        print output.get_shape().as_list()
        return output, prob