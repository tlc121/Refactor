import tensorflow as tf
import sys
sys.path.append('../')
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, deconvolutional, adaptive_pooling
from config_seg import cfg

def decoder(input_, trainable):
    with tf.variable_scope('decoder'):
        input_size = cfg.TRAIN.INPUTSIZE[0]
        out_channel = cfg.TRAIN.NUMCLASS
        pool_1 = tf.reduce_mean(input_, (1,2))[:, tf.newaxis, tf.newaxis, :]
        pool_1 = convolutional(pool_1, [1,1,pool_1.get_shape()[-1], pool_1.get_shape()[-1]/4], trainable, name='conv1')
        pool_1 = tf.image.resize_nearest_neighbor(pool_1, (input_size/8, input_size/8))

        pool_2 = adaptive_pooling(input_, input_size/8, 2)
        pool_2 = convolutional(pool_2, [1,1,pool_2.get_shape()[-1], pool_2.get_shape()[-1]/4], trainable, name='conv2')
        pool_2 = tf.image.resize_nearest_neighbor(pool_2, (input_size/8, input_size/8))

        pool_3 = adaptive_pooling(input_, input_size/8, 3)
        pool_3 = convolutional(pool_3, [1,1,pool_3.get_shape()[-1], pool_3.get_shape()[-1]/4], trainable, name='conv3')
        pool_3 = tf.image.resize_nearest_neighbor(pool_3, (input_size/8, input_size/8))

        pool_4 = adaptive_pooling(input_, input_size/8, 6)
        pool_4 = convolutional(pool_4, [1,1,pool_4.get_shape()[-1], pool_4.get_shape()[-1]/4], trainable, name='conv4')
        pool_4 = tf.image.resize_nearest_neighbor(pool_4, (input_size/8, input_size/8))

        concat_list_1 = [input_, pool_1, pool_2, pool_3, pool_4]
        concat_1 = tf.concat(concat_list_1, axis=-1)
        #concat_1 = convolutional(concat_1, [3, 3, concat_1.get_shape()[-1], concat_1.get_shape()[-1]/2], trainable, name='conv5')
        
        concat = convolutional(concat_1, [3, 3, concat_1.get_shape()[-1], out_channel], trainable, name='conv5')
        conv_5 = tf.image.resize_nearest_neighbor(concat, (input_size, input_size), name='linear')
        output = tf.nn.sigmoid(conv_5, name='op_to_store')
    
    return output
    
    
    