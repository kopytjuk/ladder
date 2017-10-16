import tensorflow as tf
from helpers import batchmean, batchstd, g
#from tensorflow.contrib.layers import dense
#from tensorflow.contrib.layers import batch_norm
#from tensorflow.nn import relu, softmax

LAYERS = [20, 10, 5]
SIGMA = 1.0
LAYER_IMPORTANCE = [0.5,0.3,0.2]

x = tf.placeholder(tf.float32, shape=(100, 28*28)) # MNIST dataset
y_gt = tf.placeholder(tf.float32, shape=(100, 100)) # MNIST dataset

# introduce noise to the inputs
x_noisy = x + tf.random_normal(shape=x.shape, mean=0.0, stddev=SIGMA, dtype=tf.float32, seed=None, name=None)

'''Encoder with noise'''
z_tilde_layers = []
h_tilde_layers = []
layer_inp = x_noisy
for i, l in enumerate(LAYERS):

    scope_name = 'L{:d}'.format(i)

    with tf.variable_scope(scope_name):
        _ = tf.layers.dense(inputs = layer_inp, units = l, activation=None, use_bias=True, reuse = None, name='dense')
        z_tilde_l += tf.random_normal(shape=_.shape, mean=0.0, stddev=SIGMA, dtype=tf.float32, seed=None, name=None)
        z_tilde_layers.append(z_tilde_l)
        out = tf.contrib.layers.batch_norm(inputs=z_tilde_l, is_training=True, scale=True, activation_fn = tf.nn.relu)
        h_tilde_layers.append(out)
        layer_inp = out

# final softmax
scope_name = 'L{:d}'.format(len(LAYERS))
with tf.variable_scope(scope_name):
    y_tilde = tf.layers.dense(inputs = out, units = 10, activation=tf.nn.softmax, use_bias=True,reuse = False) # 10 units for 10 numbers
    #prob = tf.nn.softmax(logits)

C_sv = tf.losses.softmax_cross_entropy(onehot_labels=y_gt, logits=y_tilde)

print([s.name for s in tf.global_variables()])

'''Encoder w/o noise'''
layer_inp = x
mean_vectors = []
std_vectors = []
z_list = []
for i, l in enumerate(LAYERS):
    print(i)
    scope_name = 'L{:d}'.format(i)
    with tf.variable_scope(scope_name):
        z_pre = tf.layers.dense(inputs = layer_inp, units = l, activation=None, use_bias=True, reuse = True)

        mean_vectors.append(batchmean(z_pre))
        std_vectors.append(batchstd(z_pre))

        z = tf.contrib.layers.batch_norm(inputs=z_pre, is_training=True, scale=True)
        z_list.append(z)
        h_l = tf.nn.relu(z)


        layer_inp = out

with tf.variable_scope(scope_name):
    y_pred = tf.layers.dense(inputs = out, units = 10, activation=tf.nn.softmax, use_bias=True,reuse = False)

'''Decoder'''
u_list = [y_tilde]
C_uv = [] # unsupervised cost
for i, l in enumerate(LAYERS[::-1]):
    # denoising g(z,u)

    z_hat_i = g(z_tilde_layers[-1-i], u_list[-1])
    z_hat_i_BN = tf.divide(z_hat_i - mean_vectors[-1-i], std_vectors[-1-i])
    C_uv_l = LAYER_IMPORTANCE[-1-i]*tf.reduce_mean(tf.norm(z_hat_i_BN-z_list[-1-i],axis=1),axis=0)/l
    C_uv.append(C_uv_l)

    u_list.append(tf.layers.dense(inputs = z_hat_i_BN, units = LAYER[-2-i], activation=tf.nn.softmax, use_bias=True,reuse = False))

sum = 0
for c in C_uv:
    sum += c



print([s.name for s in tf.global_variables()])
