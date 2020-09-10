import tensorflow as tf
import sys
sys.path.append('../')
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, deconvolutional, residual_dilated_block
from config_seg import cfg


scale = 1.0
def encoder(input_data, trainable):
    with tf.variable_scope('backbone'):
#         input_ = convolutional(input_data, filter_shape=[3,3,3,32], name='conv0_0', trainable=trainable)
#         input_ = convolutional(input_, filter_shape=[3,3,32,32], name='conv0_1', trainable=trainable)
#         route_0 = input_
        
#         input_ = Max_Pooing(input_, name='max_pooling_0')
#         input_ = convolutional(input_, filter_shape=[3,3,32,64], name='conv1_0', trainable=trainable)
#         convolutional(input_, filter_shape=[3,3,64,64], name='conv1_1', trainable=trainable)
#         route_1 = input_
        
#         input_ = Max_Pooing(input_, name='max_pooling_1')
#         input_ = convolutional(input_, filter_shape=[3,3,64,128], name='conv2_0', trainable=trainable)
#         input_ = convolutional(input_, filter_shape=[3,3,128,128], name='conv2_1', trainable=trainable)
#         route_2 = input_
        
#         input_ = Max_Pooing(input_, name='max_pooling_2')
#         input_ = convolutional(input_, filter_shape=[3,3,128,256], name='conv3_0', trainable=trainable)
#         input_ = convolutional(input_, filter_shape=[3,3,256,256], name='conv3_1', trainable=trainable)
#         route_3 = input_
        
#         input_ = Max_Pooing(input_, name='max_pooling_3')
#         input_ = convolutional(input_, filter_shape=[3,3,256,512], name='conv4_0', trainable=trainable)
#         input_ = convolutional(input_, filter_shape=[3,3,512,512], name='conv4_1', trainable=trainable)
#         route_4 = input_
        
        input_ = convolutional(input_data, filter_shape=[7,7,3,32/scale], name='conv1', trainable=trainable)
        input_ = convolutional(input_, filter_shape=[3,3,32,32], name='conv0_1', trainable=trainable)
        
        route_0 = input_

        input_ = residual_dilated_block(input=input_, input_channel=32/scale, filter_num1=32/scale, filter_num2=32/scale, rate=1, 
                               trainable=trainable, name='Block1')

        input_ = residual_dilated_block(input=input_, input_channel=32/scale, filter_num1=32/scale, filter_num2=32/scale, rate=1, 
                               trainable=trainable, name='Block2')

        input_ = residual_dilated_block(input=input_, input_channel=32/scale, filter_num1=64/scale, filter_num2=64/scale, rate=1, 
                               trainable=trainable, name='Block3', downsample=True)
        
        route_1 = input_

        input_ = residual_dilated_block(input=input_, input_channel=64/scale, filter_num1=64/scale, filter_num2=64/scale, rate=1, 
                               trainable=trainable, name='Block4')

        input_ = residual_dilated_block(input=input_, input_channel=64/scale, filter_num1=128/scale, filter_num2=128/scale, rate=1,
                               trainable=trainable, name='Block5', downsample=True)
        
        route_2 = input_

        input_ = residual_dilated_block(input=input_, input_channel=128/scale, filter_num1=128/scale, filter_num2=128/scale, rate=1, 
                               trainable=trainable, name='Block6')

        input_ = residual_dilated_block(input=input_, input_channel=128/scale, filter_num1=256/scale, filter_num2=256/scale, rate=1,
                               trainable=trainable, name='Block7', downsample=True)
        
        route_3 = input_
        
        input_ = residual_dilated_block(input=input_, input_channel=256/scale, filter_num1=256/scale, filter_num2=256/scale, rate=1, 
                               trainable=trainable, name='Block8')

        input_ = residual_dilated_block(input=input_, input_channel=256/scale, filter_num1=512/scale, filter_num2=512/scale, rate=1,
                               trainable=trainable, name='Block9', downsample=True)
        
        route_4 = input_
        
        input_ = residual_dilated_block(input=input_, input_channel=512/scale, filter_num1=512/scale, filter_num2=512/scale, rate=1,
                               trainable=trainable, name='Block10', downsample=True)
        
        
        return route_0, route_1, route_2, route_3, route_4, input_