import tensorflow as tf
import sys
sys.path.append('..')
from core.common import convolutional, fc_layer, denseblock, transition_block, Max_Pooing
from backbone import res34, res50, res101, res18, dense121, dense169, dense201, dense264

class Densenet169(object):
    def __init__(self, input_data, trainable, classes, keep_prob):
        self.trainable = trainable
        self.input_data = input_data
        self.num_classes = classes
        self.keep_prob = keep_prob
        self.preds, self.last_layer = self.build_network()

    def build_network(self):
        with tf.variable_scope('backbone'):
            _, _, _, input_ = dense264.dense264(self.input_data,self.trainable)

        avg_pool = tf.reduce_mean(input_, (1, 2))


        prob = fc_layer(avg_pool, name='fc_layer', trainable=self.trainable, num_classes=self.num_classes, rate=self.keep_prob)

        return prob, input_


    def compute_loss(self, labels):
        loss_val = tf.reduce_mean(-tf.reduce_sum(tf.log(tf.clip_by_value(self.preds, 1e-10, 1.0)) * labels, reduction_indices=[1]))
        correct = tf.equal(tf.argmax(self.preds, 1), tf.argmax(labels, 1))
        accurate = tf.reduce_mean(tf.cast(correct, tf.float32))
        return loss_val, accurate


    def predict(self):
        return self.preds
    
    def cam(self):
        return self.last_layer