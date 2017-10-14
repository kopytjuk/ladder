import tensorflow as tf
from tensorflow.layers import dense
from tensorflow.contrib.layers import batch_norm
from tensorflow.nn import relu, softmax

LAYERS = [20, 10, 5]
SIGMA = 1.0

x = tf.placeholder(tf.float32, shape=(None, 28*28)) # MNIST dataset

# introduce noise to the inputs
x_noisy = x + tf.random_normal(shape=x.shape, mean=0.0, stddev=SIGMA, dtype=tf.float32, seed=None, name=None)

layer_inp = x_noisy
for i, l in enumerate(LAYERS):

    scope_name = 'L{:d}'.format(i)

    with tf.variable_scope():
        _ = dense(inputs = layer_inp, units = l, activation=None, use_bias=True)
        _ += tf.random_normal(shape=_.shape, mean=0.0, stddev=SIGMA, dtype=tf.float32, seed=None, name=None)
        out = batch_norm(inputs=_, is_training=True, reuse=scope_name, scale=True, activation = relu)
        layer_inp = out

# final softmax layer
logits = dense(inputs = out, units = 10, activation=softmax, use_bias=True) # 10 units for 10 numbers
prob = softmax(logits)
