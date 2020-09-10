import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from core.common import convolutional, residual_block, Max_Pooing, fc_layer, denseblock, transition_block, Batch_Norm, RELU

def dense121(input_, trainable):
    with tf.variable_scope('backbone'):
        input_ = convolutional(input_, filter_shape=[7,7,3,64], name='conv1', trainable=trainable,
                                      downsample=True)

        input_ = Max_Pooing(input_, name='MaxPooling1')

        input_ = denseblock(input_, 6, 32, trainable= trainable, name='Denseblock1')

        input_ = transition_block(input_, 32, trainable= trainable, name='Transistion1')

        route_1 = input_

        input_ = denseblock(input_, 12, 32, trainable= trainable, name='Denseblock2')

        input_ = transition_block(input_, 32, trainable= trainable, name='Transision2')

        route_2 = input_

        input_ = denseblock(input_, 24, 32, trainable= trainable, name='Denseblock3')

        input_ = transition_block(input_, 32, trainable= trainable, name='Transision3')

        input_ = denseblock(input_, 16, 32, trainable= trainable, name='Denseblock4')

        input_ = Batch_Norm(input_, 'batch_norm', trainable)

        input_ = RELU(input_)

        route_3 = input_


        return route_1, route_2, route_3, input_
    