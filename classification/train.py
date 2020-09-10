import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from config_cls import cfg
from dataset import Dataset
from factory_cls import backbone
from random import randint
from tensorflow.python.framework import graph_util
import pandas as pd

class Classitrain(object):
    def __init__(self):
        self.Batch_Size = cfg.TRAIN.BATCHSIZE
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.num_classes = cfg.TRAIN.NUMCLASS
        self.trainset = Dataset(self.num_classes,'train')
        self.testset = Dataset(self.num_classes,'test')
        self.network = cfg.TRAIN.NETWORK
        self.train_txt = cfg.TRAIN.ANNO_PATH
        self.sess = tf.Session()
        self.model_type = cfg.TRAIN.NETWORK
        self.input_size = cfg.TRAIN.INPUTSIZE
        self.interval = cfg.TRAIN.SAVE
        self.initial_weights = cfg.TRAIN.INITIAL_WEIGHT
        self.pretrain_mode = cfg.TRAIN.PRETRAIN_MODE
        self.epoch = cfg.TRAIN.EPOCH
        self.pretrain_model = cfg.TRAIN.BACKBONE_PRETRAIN
        self.moving_ave_decay = cfg.TRAIN.MOMENTUM
        self.steps_per_period  = len(self.trainset)
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.quantity = len(open(self.train_txt, 'r').readlines())
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout')
        self.moving_ave_decay = 0.995
        self.warmup_periods = 1
        
       
        self.input_data = tf.placeholder(shape = [None, self.input_size, self.input_size, 3], dtype=tf.float32, name='input')
        self.input_labels = tf.placeholder(dtype=tf.float32, name='label')
        self.trainable = tf.placeholder(dtype=tf.bool, name='trainable')
        
        self.model = backbone(model=self.network, input_data=self.input_data, trainable=self.trainable, classes=self.num_classes, keep_prob=self.keep_prob)
        self.loss_cls, self.accurate = self.model.compute_loss(labels=self.input_labels)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.loss_cls)
        self.keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n(self.keys)
        #self.loss = self.loss_cls
        self.cam = self.model.cam()
        self.net_var = tf.global_variables()
        self.varaibles_to_restore = [var for var in self.net_var if 'backbone' in var.name]
        
            
        #moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())
        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( self.epoch * self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)
            
        #moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())
        self.optimizer =  tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=self.net_var)
        #self.optimizer = tf.train.Momen
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([self.optimizer, global_step_update]):
   #             with tf.control_dependencies([moving_ave]):
                self.train_op = tf.no_op()
        
            #self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learn_rate_init, momentum=0.8).minimize(self.loss)
        
        self.loader_backbone = tf.train.Saver(self.varaibles_to_restore)
        self.loader_whole = tf.train.Saver(tf.global_variables())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        
    def compute_keep_prob(self, now_step=0,start_value=1.0, stop_value=0.75, nr_steps=100000, trainable=False):
        prob_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)
        keep_prob = prob_values[now_step]
        return keep_prob

    def train(self):
        #self.sess.run(tf.global_variables_initializer())
        if self.pretrain_mode == 'whole':
            try:
                print ('=>Restore weights from ' + self.initial_weights)
                self.loader_whole.restore(self.sess, self.initial_weights)
            except:
                print (self.initial_weights + 'does not exist!')
                print ('=>starts training from scratch ...')
        else:
            try:
                print ('=>Restore weights from ' + self.pretrain_model)
                self.loader_backbone.restore(self.sess, self.pretrain_model)
            except Exception as e:
                print (e)
                print (self.pretrain_model + 'does not exist!')
                print ('=>starts training from scratch ...')
        
        min_loss_val = 0.8
        min_loss_train = 0.8
        i = 0
        for epoch in range(self.epoch):
            pabr = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []
            train_epoch_acc, test_epoch_acc = [], []
            for train_data in pabr:
                keep_prob = self.compute_keep_prob(now_step=epoch, nr_steps = self.epoch, trainable=True)
                _, train_step_loss, train_step_acc, _ = self.sess.run([self.train_op, self.loss, self.accurate, self.global_step], feed_dict={self.input_data: train_data[0],
                                                         self.input_labels: train_data[1],
                                                         self.trainable: True,
                                                         self.keep_prob: 1.0})
                
                train_epoch_loss.append(train_step_loss)
                train_epoch_acc.append(train_step_acc)
                pabr.set_description("train loss: %.2f" %train_step_loss)
            
            for test_data in self.testset:
                test_step_loss, test_step_acc = self.sess.run([self.loss, self.accurate],
                                                                feed_dict={self.input_data: test_data[0],
                                                                           self.input_labels: test_data[1],
                                                                           self.trainable: False, 
                                                                           self.keep_prob:1.0})

                test_epoch_loss.append(test_step_loss)
                test_epoch_acc.append(test_step_acc)

            train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = np.mean(train_epoch_loss), np.mean(train_epoch_acc), np.mean(test_epoch_loss), np.mean(test_epoch_acc)
            print ('Epoch: %2d Train loss: %.2f Train acc: %.2f'
                   %(epoch, train_epoch_loss, train_epoch_acc))

            print ('Test loss: %.2f Test acc: %.2f'
                   % (test_epoch_loss, test_epoch_acc))
            
            
            if epoch >= 10 and test_epoch_loss < min_loss_val and train_epoch_loss < min_loss_train:
                min_loss_val = test_epoch_loss
                min_loss_train = train_epoch_loss
                constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ['fc_layer/op_to_store', 'cam'])
                model_name = self.model_type+'_epoch=%d' %epoch
                ckpt_file = '/hdd/sd5/tlc/TCT/Model_ckpt/'+ model_name + '_test_loss=%.4f.ckpt' %test_epoch_loss
                #self.saver.save(self.sess, ckpt_file, global_step=epoch)
                with tf.gfile.FastGFile('/hdd/sd5/tlc/TCT/Model_pb/comb/'+model_name+'_3channel.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())


                

    def main(self):
        self.sess.run(tf.global_variables_initializer())
        self.train()
        
if __name__ == '__main__':
    Classitrain().main() 



