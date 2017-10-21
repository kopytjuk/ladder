import tensorflow as tf
import numpy as np

def batchmean(x):
    '''Calculate mean of batch'''
    return tf.reduce_mean(x, axis=0)

def batchstd(x):
    '''Calculate std of batch'''
    mean = tf.reduce_mean(x, axis=0)
    std = tf.reduce_mean(tf.square(x-mean), axis=0)
    return std

def g(z,u,l):
    u_shape = (1,l)
    a1 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a1')
    a2 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a2')
    a3 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a3')
    a4 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a4')
    a5 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a5')
    a6 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a6')
    a7 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a7')
    a8 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a8')
    a9 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a9')
    a10 = tf.Variable(initial_value=tf.random_normal(shape=u_shape), dtype=tf.float32, name='a10')

    mu_i = tf.multiply(a1,tf.nn.sigmoid(tf.multiply(a2, u) + a3)) + tf.multiply(a4, u) + a5
    v_i = tf.multiply(a6,tf.nn.sigmoid(tf.multiply(a7, u) + a8)) + tf.multiply(a9, u) + a10

    res = tf.multiply((z - mu_i), v_i) + mu_i

    return res


def to_one_hot(x, num=10):
    v = np.zeros((x.shape[0], num))
    v[np.arange(x.shape[0]), x] = 1
    return v

def uniform_layer_importance(h):

    return [1/(h+1) for i in range(h)]