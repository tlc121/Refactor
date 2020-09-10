import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer
from core.config import cfg

scale = cfg.SCALE
def darknet53(input_data, trainable):
    input_data = convolutional(input_data, filter_shape=[3, 3, 3, 32/scale], trainable=trainable, name='conv0')
    input_data = convolutional(input_data, filter_shape=[3, 3, 32/scale, 64/scale],
                                trainable=trainable, name='conv1', downsample=True)

    for i in range(1):
        input_data = residual_block(input_data, 64/scale, 32/scale, 64/scale, trainable=trainable, name='residual%d' %(i+0))

    input_data = convolutional(input_data, filter_shape=[3, 3,  64/scale, 128/scale],
                                      trainable=trainable, name='conv4', downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128/scale, 64/scale, 128/scale, trainable=trainable, name='residual%d' %(i+1))

    #modify the featuremap into strides [4, 8, 16]
    input_data = convolutional(input_data, filter_shape=[3, 3, 128/scale, 256/scale],
                                      trainable=trainable, name='conv9', downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256/scale, 128/scale, 256/scale, trainable=trainable, name='residual%d' %(i+3))

    route_1 = input_data
    #route_2 = input_data
    input_data = convolutional(input_data, filter_shape=[3, 3, 256/scale, 512/scale],
                                      trainable=trainable, name='conv26', downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512/scale, 256/scale, 512/scale, trainable=trainable, name='residual%d' %(i+11))

    route_2 = input_data
    input_data = convolutional(input_data, filter_shape=[3, 3, 512/scale, 1024/scale],
                                      trainable=trainable, name='conv43', downsample=True)
    
    route_3 = input_data
    for i in range(4):
        input_data = residual_block(input_data, 1024/scale, 512/scale, 1024/scale, trainable=trainable, name='residual%d' %(i+19))

    return route_1, route_2, input_data, route_3
