import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
from config import cfg

##########################################################################################################
def convolutional(input, filter_shape, trainable, name, downsample=False, activation=True, bn=True):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filter_shape[0] - 2) // 2 + 1, (filter_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input = tf.pad(input, paddings, 'CONSTANT')
            strides = [1,2,2,1]
            padding = 'VALID'
        else:
            strides = [1,1,1,1]
            padding = 'SAME'
        
        conv = input

        #initializer
        initial = cfg.INITIALIZER
        initial_mode = tf.random_normal_initializer(stddev=0.01)
        if initial == 'xavier':
            initial_mode = tf.contrib.layers.xavier_initializer()
        elif initial == 'uniform':
            initial_mode = tf.random_uniform_initializer()
        elif initial == 'truncated':
            intital_mode = tf.truncated_normal_initializer(stddev=0.01)
            
        #l2_reg=tf.contrib.layers.l2_regularizer(0.0001)
        weights = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=filter_shape, initializer=initial_mode)
#         bias = tf.get_variable(name='bias', shape=filter_shape[-1], trainable=True,
#                                    dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_reg(weights))
        conv = tf.nn.conv2d(input=conv, filter=weights, strides=strides, padding=padding)
        if bn:
            conv = Batch_Norm(conv, 'Batch_Norm_1', trainable)
        if activation:
            conv = RELU(conv)
        return conv

##########################################################################################################
def Batch_Norm(x, scope, training):
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(x, beta_initializer=tf.zeros_initializer(), 
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=training)
        

    
def RELU(input, name='relu'):
    with tf.variable_scope(name):
        mode = cfg.ACTIVATION
        act = tf.nn.relu(input)
        if mode == 'relu':
            act = tf.nn.relu(input)
        elif mode == 'swish':
            act = tf.nn.sigmoid(input) * input
        elif mode == 'sigmoid':
            act = tf.nn.sigmoid(input)
        elif mode == 'tanh':
            act = tf.nn.tanh(input)
        elif mode == 'leaky_relu':
            act = tf.nn.leaky_relu(input)
        elif mode == 'h-swish':
            act = (tf.nn.relu6(input + 3)/6) * input
        return act

def Max_Pooing(input, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=(1,3,3,1), strides=[1,2,2,1], padding='SAME', name='layer1')

##########################################################################################################
def fc_layer_reg(input, name, trainable, num_classes, rate, scale):
    with tf.variable_scope(name):
        dimensions = input.get_shape()
        concatenation = tf.concat([input, scale], axis=-1)
        fc_34 = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                shape=[dimensions[-1]+1, num_classes], initializer=tf.contrib.layers.xavier_initializer())
#         bias_34 = tf.get_variable(name='bias', shape=num_classes, trainable=True,
#                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
        res = tf.matmul(concatenation, fc_34)
        res= tf.nn.relu(res, name='op_to_store')
        return res
    
def fc_layer(input, name, trainable, num_classes, rate):
    with tf.variable_scope(name):
        dimensions = input.get_shape()
        fc_34 = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                shape=[dimensions[-1], num_classes], initializer=tf.contrib.layers.xavier_initializer())
#         bias_34 = tf.get_variable(name='bias', shape=num_classes, trainable=True,
#                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        res = tf.matmul(input, fc_34)
        #res = tf.nn.dropout(res, rate, name='dropout_layer')
        return tf.nn.softmax(res, name='op_to_store'), fc_34
    
def fc_layer_triplet(input, name, trainable, num_classes, rate):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        dimensions = input.get_shape()
        fc_34 = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                shape=[dimensions[-1], num_classes], initializer=tf.random_normal_initializer(stddev=0.01))
#         bias_34 = tf.get_variable(name='bias', shape=num_classes, trainable=True,
#                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        res = tf.matmul(input, fc_34)
        res = tf.nn.l2_normalize(res, 1, 1e-10, name='op_to_store')
        return res
    
##########################################################################################################
def residual_block(input, input_channel, filter_num1, filter_num2, trainable, name, downsample=False):
    with tf.variable_scope(name):
        if downsample:
            short_cut = convolutional(input, filter_shape=(3, 3, input_channel, filter_num1),
                                             trainable=trainable, bn=False,
                                             name = 'kernel_weight', downsample=downsample)
        else:
            short_cut = input

        input = convolutional(input, filter_shape=(1,1,input_channel,filter_num1), trainable=trainable, name='conv1', downsample=downsample)
        input = convolutional(input, filter_shape=(3,3,filter_num1,filter_num2), trainable=trainable, name='conv2')
        residual_output = input + short_cut
        
        return residual_output
    
##########################################################################################################    
def resblock50_101_152(input, filter_num1, filter_num2, filter_num3, trainable, name, downsample=False, FirstIn=False):
    with tf.variable_scope(name):
        shape = input.get_shape().as_list()
        input_channel = shape[-1]
        if downsample:
            short_cut = convolutional(input, filter_shape= (1, 1, input_channel, filter_num3/2),
                                       trainable=trainable, bn=False,
                                       name = 'kernel_weight', downsample=downsample)
            input = convolutional(input, filter_shape=(1, 1, input_channel, filter_num1), trainable=trainable, name='conv1', downsample=downsample)
            input = convolutional(input, filter_shape=(3, 3, filter_num1, filter_num2), trainable=trainable, name='conv2')
            input = convolutional(input, filter_shape=(1, 1, filter_num2, filter_num3/2), trainable=trainable, name='conv3')
            residual_output = input + short_cut

        else:
            if FirstIn:
                short_cut = convolutional(input, filter_shape= (1, 1, input_channel, filter_num3),
                                           trainable=trainable, bn=False,
                                           name = 'kernel_weight', downsample=downsample)
            else:
                short_cut = input
            input = convolutional(input, filter_shape=(1, 1, input_channel, filter_num1), trainable=trainable, name='conv1', downsample=downsample)
            input = convolutional(input, filter_shape=(3, 3, filter_num1, filter_num2), trainable=trainable, name='conv2')
            input = convolutional(input, filter_shape=(1, 1, filter_num2, filter_num3), trainable=trainable, name='conv3')
            residual_output = input + short_cut
        
        return residual_output
    
##########################################################################################################   
def transition_block(input, num_filters, name, trainable=True):
    with tf.variable_scope(name):
        shape = input.get_shape().as_list()
        input_shape = shape[-1]
        input = convolutional(input, filter_shape=(1, 1, input_shape, 0.5*input_shape), name='conv', trainable=trainable, bn=False)
        input = tf.nn.avg_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='avg_pool')
        return input

def Concatenation(layers):
    return tf.concat(layers, axis=3)


def bottleneck_layer(input, growth_rate, name, trainable=True):
    with tf.variable_scope(name):
        shape = input.get_shape().as_list()
        input_shape = shape[-1]
        input = convolutional(input, filter_shape=(1, 1, input_shape, 4*growth_rate), trainable=trainable, name='conv1')
        input = convolutional(input, filter_shape=(3, 3, 4*growth_rate, growth_rate), trainable=trainable, name='conv2')
        return input
    
    
    
def denseblock(input, num_layers, growth_rate, name, trainable=True):
        with tf.variable_scope(name):
            buffer = list()
            input = bottleneck_layer(input, growth_rate, trainable=trainable, name='layer0')
            buffer.append(input)
            
            for i in range(num_layers-1):
                input = Concatenation(buffer)
                input = bottleneck_layer(input, growth_rate, trainable=trainable, name='layer'+str(i+1))
                buffer.append(input)
                
            input = Concatenation(buffer)
            #input = tf.nn.dropout(input, rate, name='dropout_layer')
            return input

##########################################################################################################
def dilated_conv2d(input, kernel_size, output, name, trainable, rate=1, Activation=True):
    with tf.variable_scope(name):
        input_size = input.get_shape()[-1].value
        shape = [kernel_size, kernel_size, input_size, output]
        weights = tf.get_variable('weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        res = tf.nn.atrous_conv2d(input, weights, rate, padding='SAME')
        res = Batch_Norm(res, 'Batch_Norm', trainable)
        if Activation:
            return RELU(res)
        return res

def residual_dilated_block(input, input_channel, filter_num1, filter_num2, trainable, name, rate=2, downsample=False):
    with tf.variable_scope(name):
        if downsample:
            short_cut = convolutional(input, filter_shape=(3, 3, input_channel, filter_num1),
                                             trainable=trainable, bn=False,
                                             name = 'kernel_weight', downsample=downsample)
        else:
            short_cut = input

        input = convolutional(input, filter_shape=(3,3,input_channel,filter_num1), trainable=trainable, name='conv1', downsample=downsample)
        input = dilated_conv2d(input, kernel_size=3, output=filter_num1, name='dilated_conv1', rate=rate, Activation=True, trainable=trainable)
        input = dilated_conv2d(input, kernel_size=3, output=filter_num1, name='dilated_conv2', rate=rate, Activation=True, trainable=trainable)
        input = convolutional(input, filter_shape=(3,3,filter_num1,filter_num2), trainable=trainable, name='conv2')
        
        residual_output = input + short_cut
        residual_output = RELU(residual_output)
        return residual_output
##########################################################################################################    

    
def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        num_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, num_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output

##########################################################################################################

def deconvolutional(input_data, num_filters, kernel_size, strides, name):
    with tf.variable_scope(name):
        initial = cfg.INITIALIZER
        initial_mode = tf.random_normal_initializer(stddev=0.01)
        if initial == 'xavier':
            initial_mode = tf.contrib.layers.xavier_initializer()
        elif initial == 'uniform':
            initial_mode = tf.random_uniform_initializer()
        elif initial == 'truncated':
            intital_mode = tf.truncated_normal_initializer(stddev=0.01)
        output = tf.layers.conv2d_transpose(input_data, num_filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer = initial_mode)
        return output
    
##########################################################################################################
def depthwise_conv(input_data, filter_shape, strides, name):
    with tf.variable_scope(name):
        initial = cfg.INITIALIZER
        initial_mode = tf.random_normal_initializer(stddev=0.01)
        if initial == 'xavier':
            initial_mode = tf.contrib.layers.xavier_initializer()
        elif initial == 'uniform':
            initial_mode = tf.random_uniform_initializer()
        elif initial == 'truncated':
            intital_mode = tf.truncated_normal_initializer(stddev=0.01)
        weights = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=filter_shape, initializer=initial_mode)
        output = tf.nn.depthwise_conv2d(input_data, filter=weights, strides=strides, padding='SAME')
        return output
    
##########################################################################################################

def Mobile_V2_block(input_data, input_channel, output_channel, growth_rate, kernel_size, trainable, name, downsample=False):
    with tf.variable_scope(name):
        output = input_data
        if not downsample:
            short_cut = convolutional(input_data, filter_shape= (1, 1, input_channel, output_channel),
                                       trainable=trainable, bn=False,
                                       name = 'kernel_weight')
            
            input_ = convolutional(input_data, filter_shape= (1, 1, input_channel, input_channel*growth_rate),
                                       trainable=trainable, activation=False, 
                                       name = 'conv0')
            input_ = tf.nn.relu6(input_)
            
            input_ = depthwise_conv(input_, filter_shape=[kernel_size, kernel_size,input_channel*growth_rate,1], strides=[1,1,1,1], name='dconv0')
            
            input_ = tf.nn.relu6(input_)
            
            input_ = convolutional(input_, filter_shape= (1, 1, input_channel*growth_rate, output_channel),
                                       trainable=trainable, 
                                       name = 'conv1')
            
            output = short_cut + input_
            
        else:
            input_ = convolutional(input_data, filter_shape= (1, 1, input_channel, input_channel*growth_rate),
                                       trainable=trainable, activation=False, 
                                       name = 'conv0')
            
            input_ = tf.nn.relu6(input_)
            
            input_ = depthwise_conv(input_, filter_shape=[kernel_size, kernel_size,input_channel*growth_rate,1], strides=[1,2,2,1], name='dconv0')
            
            input_ = tf.nn.relu6(input_)
            
            input_ = convolutional(input_, filter_shape= (1, 1, input_channel*growth_rate, output_channel),
                                       trainable=trainable, 
                                       name = 'conv1')
            
            output= input_
         
        return output
            
##########################################################################################################        
def channel_shuffle(x, num_groups, name):
     with tf.variable_scope(name):
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output

##########################################################################################################

def shuffle_conv(x, input_filters, output_filters, kernel, strides , mode , name, num_groups=4):
    with tf.variable_scope(name):
        conv_side_layers_tmp = tf.split(x, num_groups ,3)
        conv_side_layers = []
        for idx, layers in enumerate(conv_side_layers_tmp):
             conv_side_layers.append(convolutional(layers, filter_shape=[kernel, kernel, input_filters//num_groups, output_filters//num_groups], strides=stirdes, name='conv_'+str(idx)))
        x = tf.concat(conv_side_layers, axis=-1)
        x = channel_shuffle(x , num_groups, 'shuffle_channel')
        return x
    
##########################################################################################################
def drop_block(input_data, keep_prob, block_size_shape, feat_size_shape):
    gamma=compute_gamma(keep_prob,feat_size_shape[1],feat_size_shape[2],block_size_shape[0],block_size_shape[1])
    shape=[feat_size_shape[0],feat_size_shape[1]-block_size_shape[0]+1,feat_size_shape[2]-block_size_shape[1]+1,feat_size_shape[3]]
    bottom = (block_size_shape[0]-1) // 2
    right = (block_size_shape[1]-1) // 2
 
    top = (block_size_shape[0]-1) // 2
    left = (block_size_shape[1]-1) // 2
 
    padding = [[0, 0], [top, bottom], [left, right], [0, 0]]
    mask=compute_block_mask(shape,padding,gamma,block_size_shape)
    
    normalize_mask=mask*tf.to_float(tf.size(mask)) / tf.reduce_sum(mask)
    outputs=input_data*normalize_mask
    return outputs


def compute_gamma(keep_prob,feat_size_h,feat_size_w,block_size_h,block_size_w):
    feat_size_h=tf.to_float(feat_size_h)
    feat_size_w=tf.to_float(feat_size_w)
    gamma=(1-keep_prob)*(feat_size_h*feat_size_w)/(block_size_h*block_size_w)/((feat_size_h-block_size_h+1)*(feat_size_w-block_size_w+1))
    return gamma
 
def bernoulli(shape,gamma=0):
    mask=tf.cast(tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)<gamma,tf.float32)
    return mask
 
def compute_block_mask(shape,padding,gamma,block_size_shape):
    mask = bernoulli(shape,gamma)
    mask = tf.pad(mask, padding,"CONSTANT")
    mask = tf.nn.max_pool(mask, [1,block_size_shape[0],block_size_shape[1], 1], [1, 1, 1, 1], 'SAME')
    mask = 1 - mask
    return mask
##########################################################################################################

def adaptive_pooling(input_data, inputsz, outputsz):
    stridesz = np.floor(inputsz/outputsz).astype(np.int32)
    kernelsz = inputsz-(outputsz-1)*stridesz
    ret = tf.nn.avg_pool(input_data, ksize=[1, kernelsz, kernelsz, 1], strides=[1, stridesz, stridesz, 1], padding='SAME', name='pool')
    return ret
    
    
    