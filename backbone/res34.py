import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer
from core.config import cfg

scale = cfg.SCALE

def Res34(input_, trainable):
    
    ori_img = input_
    
    input_ = convolutional(input_, filter_shape=[7,7,3,64/scale], name='conv1', trainable=trainable,
                                  downsample=True)
    
    route_0 = input_

    input_ = Max_Pooing(input_, name='MaxPooling1')
    
    route_1 = input_

    input_ = residual_block(input=input_, input_channel=64/scale, filter_num1=64/scale, filter_num2=64/scale,
                           trainable=trainable, name='Block1')

    input_ = residual_block(input=input_, input_channel=64/scale, filter_num1=64/scale, filter_num2=64/scale,
                           trainable=trainable, name='Block2')

    input_ = residual_block(input=input_, input_channel=64/scale, filter_num1=64/scale, filter_num2=64/scale,
                           trainable=trainable, name='Block3')

    input_ = residual_block(input=input_, input_channel=64/scale, filter_num1=128/scale, filter_num2=128/scale,
                           trainable=trainable, name='Block4', downsample=True)
    
    route_2 = input_

    input_ = residual_block(input=input_, input_channel=128/scale, filter_num1=128/scale, filter_num2=128/scale,
                           trainable=trainable, name='Block5')

    input_ = residual_block(input=input_, input_channel=128/scale, filter_num1=128/scale, filter_num2=128/scale,
                           trainable=trainable, name='Block6')

    input_ = residual_block(input=input_, input_channel=128/scale, filter_num1=128/scale, filter_num2=128/scale,
                           trainable=trainable, name='Block7')

    input_ = residual_block(input=input_, input_channel=128/scale, filter_num1=256/scale, filter_num2=256/scale,
                           trainable=trainable, name='Block8', downsample=True)

    route_3 = input_
    
    input_ = residual_block(input=input_, input_channel=256/scale, filter_num1=256/scale, filter_num2=256/scale,
                           trainable=trainable, name='Block9')

    input_ = residual_block(input=input_, input_channel=256/scale, filter_num1=256/scale, filter_num2=256/scale,
                           trainable=trainable, name='Block10')

    input_ = residual_block(input=input_, input_channel=256/scale, filter_num1=256/scale, filter_num2=256/scale,
                           trainable=trainable, name='Block11')

    input_ = residual_block(input=input_, input_channel=256/scale, filter_num1=256/scale, filter_num2=256/scale,
                           trainable=trainable, name='Block12')

    input_ = residual_block(input=input_, input_channel=256/scale, filter_num1=256/scale, filter_num2=256/scale,
                           trainable=trainable, name='Block13')

    input_ = residual_block(input=input_, input_channel=256/scale, filter_num1=512/scale, filter_num2=512/scale,
                           trainable=trainable, name='Block14', downsample=True)

    route_4 = input_
    
    input_ = residual_block(input=input_, input_channel=512/scale, filter_num1=512/scale, filter_num2=512/scale,
                           trainable=trainable, name='Block15')

    input_ = residual_block(input=input_, input_channel=512/scale, filter_num1=512/scale, filter_num2=512/scale,
                           trainable=trainable, name='Block16')
    
    return ori_img, route_0, route_1, route_2, route_3, route_4, input_
    