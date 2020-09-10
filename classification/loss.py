import tensorflow as tf

def Cross_entropy(pred, label):
    loss_val = tf.reduce_mean(-tf.reduce_sum(tf.log(tf.clip_by_value(pred,1e-10,1.0)) * label, reduction_indices=[1]))
    #loss_val = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits=pred), axis=[1]))
    return loss_val



def MSE(pred, label):
    loss_val = tf.reduce_mean(tf.square(pred - label))
    return loss_val