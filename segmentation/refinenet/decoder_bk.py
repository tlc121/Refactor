import tensorflow as tf
import sys
sys.path.append('../')
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, deconvolutional
from config_seg import cfg


def Max_Pooling(input, size, stride, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=(1,size,size,1), strides=[1,stride,stride,1], padding='SAME', name='layer1')

def decoder(route_0, route_1, route_2, route_3, route_4, trainable):
    with tf.variable_scope('decoder'):
        out_channel = cfg.TRAIN.NUMCLASS
        
        #first refinenet
        route_3 = residual_block(route_3, route_3.get_shape()[-1], route_3.get_shape()[-1], route_3.get_shape()[-1], trainable, name = 'rcu1')
        route_4 = residual_block(route_4, route_4.get_shape()[-1], route_4.get_shape()[-1], route_4.get_shape()[-1], trainable, name = 'rcu2')
        up_4= deconvolutional(route_4, route_4.get_shape()[-1]/2, 2, (2,2), name='deconv1')
        fusion3 = tf.nn.relu(route_3 + up_4) #fusion3 is a 8 times downsamples
        
        
        mp1 = Max_Pooling(fusion3, 3, 2, name='pool1')
        mp1 = convolutional(mp1, filter_shape=[3,3, mp1.get_shape()[-1], mp1.get_shape()[-1]], name='conv1_0', trainable=trainable)
        mp1_deconv = deconvolutional(mp1, mp1.get_shape()[-1], 2, (2,2), name='deconv0_1')
        
        mp2 = Max_Pooling(mp1, 3, 2, name='pool2')
        mp2 = convolutional(mp2, filter_shape=[3,3, mp2.get_shape()[-1], mp2.get_shape()[-1]], name='conv1_1', trainable=trainable)
        mp2_deconv = deconvolutional(mp2, mp2.get_shape()[-1], 2, (4,4), name='deconv0_2')
        
        mp3 = Max_Pooling(mp2, 3, 2, name='pool3')
        mp3 = convolutional(mp3, filter_shape=[3,3, mp3.get_shape()[-1], mp3.get_shape()[-1]], name='conv1_2', trainable=trainable)
        mp3_deconv = deconvolutional(mp3, mp3.get_shape()[-1], 2, (8,8), name='deconv0_3')
        
        output_refine_1 = fusion3 + mp1_deconv + mp2_deconv + mp3_deconv
        output_refine_1 = residual_block(output_refine_1, output_refine_1.get_shape()[-1], output_refine_1.get_shape()[-1], output_refine_1.get_shape()[-1], trainable, name = 'rcul0')
        
        
        #second refinenet
        route_1 = residual_block(route_1, route_1.get_shape()[-1], route_1.get_shape()[-1], route_1.get_shape()[-1], trainable, name = 'rcu3')
        route_2 = residual_block(route_2, route_2.get_shape()[-1], route_2.get_shape()[-1], route_2.get_shape()[-1], trainable, name = 'rcu4')
        fusion3 = residual_block(output_refine_1, output_refine_1.get_shape()[-1], output_refine_1.get_shape()[-1], output_refine_1.get_shape()[-1], trainable, name = 'rcu5')
        up_3 = deconvolutional(fusion3, fusion3.get_shape()[-1]/2, 2, (2,2), name='deconv2')
        fusion4 = route_2 + up_3 #fusion4 is a 4 times downsamples
        up_2 = deconvolutional(fusion4, fusion4.get_shape()[-1]/2, 2, (2,2), name='deconv3')
        fusion5 = tf.nn.relu(up_2 + route_1)
        
        mp4 = Max_Pooling(fusion5, 3, 2, name='pool4')
        mp4 = convolutional(mp4, filter_shape=[3,3, mp4.get_shape()[-1], mp4.get_shape()[-1]], name='conv2_0', trainable=trainable)
        mp4_deconv = deconvolutional(mp4, mp4.get_shape()[-1], 2, (2,2), name='deconv1_1')
        
        mp5 = Max_Pooling(mp4, 3, 2, name='pool5')
        mp5 = convolutional(mp5, filter_shape=[3,3, mp5.get_shape()[-1], mp5.get_shape()[-1]], name='conv2_1', trainable=trainable)
        mp5_deconv = deconvolutional(mp5, mp5.get_shape()[-1], 2, (4,4), name='deconv1_2')
        
        mp6 = Max_Pooling(mp5, 3, 2, name='pool6')
        mp6 = convolutional(mp6, filter_shape=[3,3, mp6.get_shape()[-1], mp6.get_shape()[-1]], name='conv2_2', trainable=trainable)
        mp6_deconv = deconvolutional(mp6, mp6.get_shape()[-1], 2, (8,8), name='deconv1_3')
        
        output_refine_2 = fusion5 + mp4_deconv + mp5_deconv + mp6_deconv
        output_refine_2 = residual_block(output_refine_2, output_refine_2.get_shape()[-1], output_refine_2.get_shape()[-1], output_refine_2.get_shape()[-1], trainable, name = 'rcul1')
        
        output_refine_2 = deconvolutional(output_refine_2, output_refine_2.get_shape()[-1]/2, 2, (2,2), name='deconv1_4')
        conv_5 = convolutional(output_refine_2, filter_shape=[3,3, output_refine_2.get_shape()[-1], out_channel], name='conv5', trainable=trainable)
        output = tf.nn.sigmoid(conv_5, name='op_to_store')
        print output.get_shape().as_list
        return output