import tensorflow as tf

def batchmean(x):
    '''Calculate mean of batch'''
    return tf.reduce_mean(x, axis=0)

def batchstd(x):
    '''Calculate std of batch'''
    mean = tf.reduce_mean(x, axis=0)
    std = tf.reduce_mean(tf.square(x-mean), axis=0)
    return std
