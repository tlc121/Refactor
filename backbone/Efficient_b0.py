import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, depthwise_conv, Mobile_V2_block
from core.config import cfg

scale = cfg.SCALE

def Efficient_B0(input_, trainable):
    
    ori_img = input_
    
    input_ = convolutional(input_, filter_shape=[3,3,3,32/scale], trainable=trainable, name='conv1', downsample=True)
    
    route_0 = input_
    
    input_ = Mobile_V2_block(input_, 32/scale, 16/scale, kernel_size=3, growth_rate=1, trainable=trainable, name='block1')
    
    input_ = Mobile_V2_block(input_, 16/scale, 24/scale, kernel_size=3, growth_rate=6, trainable=trainable, name='block2')
    
    input_ = Mobile_V2_block(input_, 24/scale, 24/scale, kernel_size=3, growth_rate=6, trainable=trainable, name='block3', downsample=True)
    
    route_1 = input_
    
    input_ = Mobile_V2_block(input_, 24/scale, 40/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block4')
    
    input_ = Mobile_V2_block(input_, 40/scale, 40/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block5', downsample=True)
    
    route_2 = input_
    
    input_ = Mobile_V2_block(input_, 40/scale, 80/scale, kernel_size=3, growth_rate=6, trainable=trainable, name='block6')
    
    input_ = Mobile_V2_block(input_, 80/scale, 80/scale, kernel_size=3, growth_rate=6, trainable=trainable, name='block7')
    
    input_ = Mobile_V2_block(input_, 80/scale, 80/scale, kernel_size=3, growth_rate=6, trainable=trainable, name='block8', downsample=True)
    
    route_3 = input_
    
    input_ = Mobile_V2_block(input_, 80/scale, 112/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block11')
    
    input_ = Mobile_V2_block(input_, 112/scale, 112/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block12')
    
    input_ = Mobile_V2_block(input_, 112/scale, 112/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block13')
    
    
    input_ = Mobile_V2_block(input_, 112/scale, 192/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block14')
    
    input_ = Mobile_V2_block(input_, 192/scale, 192/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block15')
    
    input_ = Mobile_V2_block(input_, 192/scale, 192/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block16', downsample=True)
    
    input_ = Mobile_V2_block(input_, 192/scale, 320/scale, kernel_size=5, growth_rate=6, trainable=trainable, name='block17')
    
    input_ = convolutional(input_, filter_shape=[1,1,320/scale, 1280/scale], trainable=trainable, name='conv2')
    
    route_4 = input_
    
    return ori_img, route_0, route_1, route_2, route_3, route_4, input_


