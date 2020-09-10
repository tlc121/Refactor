import tensorflow as tf
import numpy as np
from config_seg import cfg



def CrossEntropy_Loss(pred, label):
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
    ce_loss = tf.reduce_mean(tf.reduce_mean(ce_loss,axis=[1,2,3]))
    return ce_loss


def dice_loss(pred, label):
    num_classes = cfg.TRAIN.NUMCLASS
    batch_size = cfg.TRAIN.BATCHSIZE
    smooth = 1e-5
    total_loss = 0
    for i in range(num_classes):
        intersection = tf.reduce_sum(pred[..., i:i+1]*label[..., i:i+1], axis=[1,2,3])
        pred_sum = tf.reduce_sum(pred[..., i:i+1]*pred[..., i:i+1], axis=[1,2,3])
        label_sum = tf.reduce_sum(label[..., i:i+1]*label[..., i:i+1], axis=[1,2,3])
        total_loss += tf.reduce_mean(1- ((2. * intersection) / (pred_sum + label_sum + smooth)))
    return total_loss / num_classes

# def dice_loss_head(pred, label):
#     num_classes = cfg.TRAIN.NUMCLASS
#     batch_size = cfg.TRAIN.BATCHSIZE
#     smooth = 1e-5
#     total_loss = 0
#     for i in range(num_classes):
#         intersection = tf.reduce_sum(pred[..., i:i+1]*label[..., i:i+1], axis=[1,2,3])
#         pred_sum = tf.reduce_sum(pred[..., i:i+1]*pred[..., i:i+1], axis=[1,2,3])
#         label_sum = tf.reduce_sum(label[..., i:i+1]*label[..., i:i+1], axis=[1,2,3])
#         loss_b = 0
#         for j in range(batch_size):
#             loss = tf.cond(pred=label_sum[j, ...] > 0.0,
#                  true_fn=lambda: 1- ((2. * intersection[j, ...]) / (pred_sum[j, ...] + label_sum[j, ...] + smooth)),
#                  false_fn=lambda: 0.0
#                    )
#             loss_b += loss
#         total_loss += loss_b/batch_size #tf.reduce_mean(1- ((2. * intersection) / (pred_sum + label_sum + smooth)))
#     return total_loss / num_classes