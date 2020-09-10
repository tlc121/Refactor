import tensorflow as tf

def Cross_entropy(pred, label):
    loss_val = tf.reduce_mean(-tf.reduce_sum(tf.log(pred) * label, reduction_indices=[1]))
    return loss_val



def MSE(pred, label):
    #loss_val = tf.reduce_mean(tf.square(pred - label))
    loss_val = tf.losses.mean_squared_error(label, pred)
    return loss_val