import tensorflow as tf

def batchmean(x):
    '''Calculate mean of batch'''
    return tf.reduce_mean(x, axis=0)

def batchstd(x):
    '''Calculate std of batch'''
    mean = tf.reduce_mean(x, axis=0)
    std = tf.reduce_mean(tf.square(x-mean), axis=0)
    return std

def g(z,u):

    a1 = tf.Variable(initial_value=tf.random_normal(shape=z.shape), dtype=tf.float32)
    a2 = tf.Variable(initial_value=tf.random_normal(shape=z.shape), dtype=tf.float32)
    a3 = tf.Variable(initial_value=tf.random_normal(shape=z.shape), dtype=tf.float32)
    a4 = tf.Variable(initial_value=tf.random_normal(shape=z.shape), dtype=tf.float32)
    a5 = tf.Variable(initial_value=tf.random_normal(shape=z.shape), dtype=tf.float32)
    a6 = tf.Variable(initial_value=tf.random_normal(shape=u.shape), dtype=tf.float32)
    a7 = tf.Variable(initial_value=tf.random_normal(shape=u.shape), dtype=tf.float32)
    a8 = tf.Variable(initial_value=tf.random_normal(shape=u.shape), dtype=tf.float32)
    a9 = tf.Variable(initial_value=tf.random_normal(shape=u.shape), dtype=tf.float32)
    a10 = tf.Variable(initial_value=tf.random_normal(shape=u.shape), dtype=tf.float32)

    mu_i = a1*tf.nn.sigmoid(tf.mul(a2, z) + a3) + tf.mul(a4, z) + a5
    v_i = a6*tf.nn.sigmoid(tf.mul(a7, z) + a8) + tf.mul(a9, z) + a10

    res = tf.mul((z - mu_i),v_i) + mu_i

    return res
