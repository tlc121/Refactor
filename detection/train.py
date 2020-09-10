import os
import time
import shutil
import numpy as np
import tensorflow as tf
import utils as utils
from tqdm import tqdm
from dataset import Dataset
from one_stage.yolov3 import YOLOV3
from config_det import cfg
from tensorflow.python.framework import graph_util
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.pretrain_mode       = cfg.TRAIN.PRETRAIN_MODE
        self.model_backbone      = cfg.TRAIN.BACKBONE
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./log/train"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.pretrain_model      = cfg.TRAIN.BACKBONE_PRETRAIN
        self.steps_per_period    = len(self.trainset)
        self.save_pb_path        = cfg.TRAIN.SAVE_PATH_PB
        self.save_ckpt_path      = cfg.TRAIN.SAVE_PATH_CKPT
        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        
        self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
        self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
        self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
        self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
        self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
        self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
        self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
        self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

       
        self.model = YOLOV3(self.input_data, self.trainable)
        self.net_var = tf.global_variables()
        self.varaibles_to_restore = [var for var in self.net_var if 'backbone' in var.name]
        self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
        self.loss = self.giou_loss + self.conf_loss + self.prob_loss
        self.net_var = tf.global_variables()
        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

#         with tf.name_scope("define_weight_decay"):
#             moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())
            
        
        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    #with tf.control_dependencies([moving_ave]):
                    self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    #with tf.control_dependencies([moving_ave]):
                    self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader_all = tf.train.Saver(self.net_var)
            self.loader_backbone = tf.train.Saver(self.varaibles_to_restore)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)


    def train(self):
        self.sess.run(tf.global_variables_initializer())
        
        #different pretrain models
        if self.pretrain_mode == 'whole':
            try:
                print('=> Restoring weights from: %s ... ' % self.initial_weight)
                self.loader_all.restore(self.sess, self.initial_weight)
            except Exception as e:
                print e
                print('=> %s does not exist !!!' % self.initial_weight)
                print('=> Now it starts to train YOLOV3 from scratch ...')
        else:
            try:
                print('=> Restoring weights from: %s ... ' % self.initial_weight)
                self.loader_backbone.restore(self.sess, self.pretrain_model)
            except Exception as e:
                print e
                print('=> %s does not exist !!!' % self.pretrain_model)
                print('=> Now it starts to train YOLOV3 from scratch ...')
                
        min_loss = 999
        self.first_stage_epochs = 0
        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            for test_data in self.testset:
                test_step_loss = self.sess.run( self.loss, feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label_sbbox:  test_data[1],
                                                self.label_mbbox:  test_data[2],
                                                self.label_lbbox:  test_data[3],
                                                self.true_sbboxes: test_data[4],
                                                self.true_mbboxes: test_data[5],
                                                self.true_lbboxes: test_data[6],
                                                self.trainable:    False,
                })

                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            print("=> Epoch: %2d  Train loss: %.2f Test loss: %.2f"
                                %(epoch,train_epoch_loss, test_epoch_loss))
            
            if epoch % 5 == 0 and epoch >= 5: #and test_epoch_loss <= min_loss:
                min_loss = test_epoch_loss
                model_name = self.model_backbone + '_test_loss=%.4f' %test_epoch_loss
                ckpt_file = self.save_ckpt_path + model_name
                log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                                %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
                self.saver.save(self.sess, ckpt_file, global_step=epoch)
                constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ['pred_sbbox/op_to_store', 'pred_mbbox/op_to_store', 'pred_lbbox/op_to_store'])
                with tf.gfile.FastGFile(self.save_pb_path + model_name+'.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())



if __name__ == '__main__': YoloTrain().train()