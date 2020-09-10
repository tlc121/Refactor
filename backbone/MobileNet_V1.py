import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, depthwise_conv
from core.config import cfg

scale = cfg.SCALE

def MobileNet_V1(input_, trainable):
    
    ori_img = input_
    
    input_ = convolutional(input_, filter_shape=[3,3,3,32/scale], trainable=trainable, name='conv1', downsample=True)
    
    route_0 = input_
    
    input_ = depthwise_conv(input_, filter_shape=[3,3,32/scale,1], strides=[1,1,1,1], name='dw_conv1')
    input_ = convolutional(input_, filter_shape=[1, 1, 32/scale, 64/scale], trainable=trainable, name='pw_conv1')
    
    
    input_ = depthwise_conv(input_, filter_shape=[3,3,64/scale,1], strides=[1,2,2,1], name='dw_conv2')
    input_ = convolutional(input_, filter_shape=[1, 1, 64/scale, 128/scale], trainable=trainable, name='pw_conv2')
    
    route_1 = input_
    
    input_ = depthwise_conv(input_, filter_shape=[3,3,128/scale,1], strides=[1,1,1,1], name='dw_conv3')
    input_ = convolutional(input_, filter_shape=[1, 1, 128/scale, 128/scale], trainable=trainable, name='pw_conv3')
    
    input_ = depthwise_conv(input_, filter_shape=[3,3,128/scale,1], strides=[1,2,2,1], name='dw_conv4')
    input_ = convolutional(input_, filter_shape=[1, 1, 128/scale, 256/scale], trainable=trainable, name='pw_conv4')
    
    route_2 = input_
    
    
    input_ = depthwise_conv(input_, filter_shape=[3,3,256/scale,1], strides=[1,1,1,1], name='dw_conv5')
    input_ = convolutional(input_, filter_shape=[1, 1, 256/scale, 256/scale], trainable=trainable, name='pw_conv5')
    
    input_ = depthwise_conv(input_, filter_shape=[3,3,256/scale,1], strides=[1,2,2,1], name='dw_conv6')
    input_ = convolutional(input_, filter_shape=[1, 1, 256/scale, 512/scale], trainable=trainable, name='pw_conv6')
    
    route_3 = input_
    
    for i in range(5):
        input_ = depthwise_conv(input_, filter_shape=[3,3,512/scale,1], strides=[1,1,1,1], name='dw_conv' + str(7+i))
        input_ = convolutional(input_, filter_shape=[1, 1, 512/scale, 512/scale], trainable=trainable, name='pw_conv'+ str(7+i))
        
    
    input_ = depthwise_conv(input_, filter_shape=[3,3,512/scale,1], strides=[1,2,2,1], name='dw_conv12')
    input_ = convolutional(input_, filter_shape=[1, 1, 512/scale, 1024/scale], trainable=trainable, name='pw_conv12')  
    
    route_4 =input_
    
    return ori_img, route_0, route_1, route_2, route_3, route_4, input_
    
    