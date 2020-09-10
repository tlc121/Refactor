import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer

def Res50(input_, trainable):
    ori_img = input_
    
    input_ = convolutional(input_, filter_shape=(7,7,3,64), name='conv1', trainable=trainable,
                              downsample=True, bn=False)
    route_0 = input_
    
    input_ = Max_Pooing(input_, name='MaxPooling1')

    route_1 = input_
    
    for i in range(3):
        if i == 0:
            input_ = resblock50_101_152(input_, filter_num1=64, filter_num2=64, filter_num3=256,
                                        trainable=trainable, name='Block%d' % (i + 1), FirstIn=True)
            
        elif i < 2:
            input_ = resblock50_101_152(input_, filter_num1=64, filter_num2=64, filter_num3=256,
                                        trainable=trainable, name='Block%d' % (i + 1))
        else:
            input_ = resblock50_101_152(input_, filter_num1=64, filter_num2=64, filter_num3=256,
                                        trainable=trainable, name='Block%d' % (i + 1), downsample=True)
    route_2 = input_

    for i in range(4):
        if i == 0:
            input_ = resblock50_101_152(input_, filter_num1=128, filter_num2=128, filter_num3=512,
                                        trainable=trainable, name='Block%d' % (i + 1 + 3), FirstIn=True)
        elif i < 3:
            input_ = resblock50_101_152(input_, filter_num1=128, filter_num2=128, filter_num3=512,
                                        trainable=trainable, name='Block%d' % (i + 1 + 3))
        else:
            input_ = resblock50_101_152(input_, filter_num1=128, filter_num2=128, filter_num3=512,
                                        trainable=trainable, name='Block%d' % (i + 1 + 3), downsample=True)
            
    route_3 = input_
    for i in range(6):
        if i == 0:
            input_ = resblock50_101_152(input_, filter_num1=256, filter_num2=256, filter_num3=1024,
                                        trainable=trainable, name='Block%d' % (i + 1 + 7), FirstIn=True)
        elif i < 5:
            input_ = resblock50_101_152(input_, filter_num1=256, filter_num2=256, filter_num3=1024,
                                        trainable=trainable, name='Block%d' % (i + 1 + 7))

        else:
            input_ = resblock50_101_152(input_, filter_num1=256, filter_num2=256, filter_num3=1024,
                                        trainable=trainable, name='Block%d' % (i + 1 + 7), downsample=True)
    
    route_4 = input_
    for i in range(3):
        if i == 0:
            input_ = resblock50_101_152(input_, filter_num1=512, filter_num2=512, filter_num3=2048,
                               trainable=trainable, name='Block%d' % (i + 1 + 13), FirstIn=True) 
        else:
            input_ = resblock50_101_152(input_, filter_num1=512, filter_num2=512, filter_num3=2048,
                               trainable=trainable, name='Block%d' % (i + 1 + 13))
    return ori_img, route_0, route_1, route_2, route_3, route_4, input_