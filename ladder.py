import tensorflow as tf
from helpers import batchmean, batchstd, g

LAYERS = [20, 10, 10]
SIGMA = 1.0
LAYER_IMPORTANCE = [0.5,0.3,0.2]

x = tf.placeholder(tf.float32, shape=(None, 28*28)) # MNIST dataset
y_gt = tf.placeholder(tf.float32, shape=(None, 10)) # MNIST dataset

# introduce noise to the inputs
x_noisy = x + tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=SIGMA, dtype=tf.float32, seed=None, name='n_0')

'''Encoder with noise'''
z_tilde_layers = []
h_tilde_layers = []
layer_inp = x_noisy
for i, l in enumerate(LAYERS):

    scope_name = 'L{:d}'.format(i)

    with tf.variable_scope(scope_name):
        z_tilde_l = tf.layers.dense(inputs = layer_inp, units = l, activation=None, use_bias=True, reuse = None)
        z_tilde_l += tf.random_normal(shape=tf.shape(z_tilde_l), mean=0.0, stddev=SIGMA, dtype=tf.float32, seed=None, name=None)
        z_tilde_layers.append(z_tilde_l)
        if i < (len(LAYERS)-1):
            out = tf.contrib.layers.batch_norm(inputs=z_tilde_l, is_training=True, scale=True, activation_fn = tf.nn.relu)
        else:
            out = tf.contrib.layers.batch_norm(inputs=z_tilde_l, is_training=True, scale=True, activation_fn=tf.nn.softmax)
        h_tilde_layers.append(out)
        layer_inp = out

y_tilde = out

# define supervised loss
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
    with tf.variable_scope(scope_name, reuse=True):
        z_pre = tf.layers.dense(inputs = layer_inp, units = l, activation=None, use_bias=True, reuse = True)

        mean_vectors.append(batchmean(z_pre))
        std_vectors.append(batchstd(z_pre))

        z = tf.contrib.layers.batch_norm(inputs=z_pre, is_training=True, scale=True)
        z_list.append(z)

        if i < (len(LAYERS)-1):
            h_l = tf.nn.relu(z)
        else:
            h_l = tf.nn.softmax(z)

        layer_inp = h_l

'''Decoder'''
u_list = [y_tilde]
C_uv = [] # unsupervised cost
for i, l in enumerate(LAYERS[::-1]):
    # denoising g(z,u)

    L_nr = str(len(LAYERS)-i)

    if i==0:
        u_l = tf.contrib.layers.batch_norm(inputs=u_list[-1-i], is_training=True, scale=True)
        print('')
        z_hat_i = g(z_tilde_layers[-1-i], u_l, l)
    else:
        _ = tf.layers.dense(inputs=u_list[-1-i], units=l, name='V_'+L_nr)
        u_l = tf.contrib.layers.batch_norm(inputs=_, is_training=True, scale=True)
        z_hat_i = g(z_tilde_layers[-1-i], u_l, l)

    z_hat_i_BN = tf.divide(z_hat_i - mean_vectors[-1-i], std_vectors[-1-i])

    C_uv_l = LAYER_IMPORTANCE[-1-i]*tf.reduce_mean(tf.norm(z_hat_i_BN-z_list[-1-i],axis=1),axis=0)/l
    C_uv.append(C_uv_l)

    u_list.append(z_hat_i_BN)

C_uv_sum = 0
for c in C_uv:
    C_uv_sum += c

C_result = C_uv_sum + C_uv


print([s.name for s in tf.global_variables()])
