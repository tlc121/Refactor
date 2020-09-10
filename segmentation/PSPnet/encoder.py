import tensorflow as tf
import sys
sys.path.append('../')
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, deconvolutional, residual_dilated_block
from backbone import res34, res50, res101, res18, res152
from config_seg import cfg


scale = 2.0
def encoder(input_data, trainable):
    with tf.variable_scope('backbone'):
        input_ = convolutional(input_data, filter_shape=[7,7,3,64/scale], name='conv1', trainable=trainable)


        input_ = residual_dilated_block(input=input_, input_channel=64/scale, filter_num1=64/scale, filter_num2=64/scale, rate=1, 
                               trainable=trainable, name='Block1')

        input_ = residual_dilated_block(input=input_, input_channel=64/scale, filter_num1=64/scale, filter_num2=64/scale, rate=1, 
                               trainable=trainable, name='Block2')

        input_ = residual_dilated_block(input=input_, input_channel=64/scale, filter_num1=128/scale, filter_num2=128/scale, rate=1, 
                               trainable=trainable, name='Block3', downsample=True)
        
        route_1 = input_

        input_ = residual_dilated_block(input=input_, input_channel=128/scale, filter_num1=128/scale, filter_num2=128/scale, rate=1, 
                               trainable=trainable, name='Block4')

        input_ = residual_dilated_block(input=input_, input_channel=128/scale, filter_num1=256/scale, filter_num2=256/scale, rate=2,
                               trainable=trainable, name='Block5', downsample=True)
        
        route_2 = input_

        input_ = residual_dilated_block(input=input_, input_channel=256/scale, filter_num1=256/scale, filter_num2=256/scale, rate=1, 
                               trainable=trainable, name='Block6')

        route_3 = residual_dilated_block(input=input_, input_channel=256/scale, filter_num1=512/scale, filter_num2=512/scale, rate=4,
                               trainable=trainable, name='Block7', downsample=True)
        
        return route_3