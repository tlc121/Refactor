import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer
from core.config import cfg

scale = cfg.SCALE

def Res152(input_, trainable):
    ori_img = input_
    
    input_ = convolutional(input_, filter_shape=[7,7,3,64/scale], name='conv1', trainable=trainable,
                                  downsample=True)
    route_0 = input_

    input_ = Max_Pooing(input_, name='MaxPooling1')
    
    route_1 = input_

    for i in range(3):
        if i == 0:
            input_ = resblock50_101_152(input_, filter_num1=64/scale, filter_num2=64/scale, filter_num3=256/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1), FirstIn=True)
        elif i < 2:
            input_ = resblock50_101_152(input_, filter_num1=64/scale, filter_num2=64/scale, filter_num3=256/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1))
        else:
            input_ = resblock50_101_152(input_, filter_num1=64/scale, filter_num2=64/scale, filter_num3=256/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1), downsample=True)
    
    route_2 = input_
    
    for i in range(8):
        if i == 0:
            input_ = resblock50_101_152(input_, filter_num1=128/scale, filter_num2=128/scale, filter_num3=512/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1 + 3),FirstIn=True)
        elif i < 7:
            input_ = resblock50_101_152(input_, filter_num1=128/scale, filter_num2=128/scale, filter_num3=512/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1 + 3))
        else:
            input_ = resblock50_101_152(input_, filter_num1=128/scale, filter_num2=128/scale, filter_num3=512/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1 + 3), downsample=True)
    
    route_3 = input_
    
    for i in range(36):
        if i == 0:
            input_ = resblock50_101_152(input_, filter_num1=256/scale, filter_num2=256/scale, filter_num3=1024/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1 + 11), FirstIn=True)
        elif i < 35:
            input_ = resblock50_101_152(input_, filter_num1=256/scale, filter_num2=256/scale, filter_num3=1024/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1 + 11))

        else:
            input_ = resblock50_101_152(input_, filter_num1=256/scale, filter_num2=256/scale, filter_num3=1024/scale,
                                        trainable=self.trainable, name='Block%d' % (i + 1 + 11), downsample=True)
    
    route_4 = input_
    
    for i in range(3):
        if i == 0:
            input_ = resblock50_101_152(input_, filter_num1=512/scale, filter_num2=512/scale, filter_num3=2048/scale,trainable=self.trainable, name='Block%d' % (i + 1 + 47), FirstIn=True)
        else:
            input_ = resblock50_101_152(input_, filter_num1=512/scale, filter_num2=512/scale, filter_num3=2048/scale,trainable=self.trainable, name='Block%d' % (i + 1 + 47))

    
    return ori_img, route_0, route_1, route_2, route_3, route_4, input_
    