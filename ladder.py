import tensorflow as tf
from helpers import batchmean, batchstd, g


def model_fn(features, labels, mode, params):

    param_lr = params['learning_rate']
    param_layers = params['layer_def']
    param_importance = params['layer_importance']
    param_device = params['device'] # cpu or gpu
    param_n_sigma = params['n_sigma'] # noise value

    x = features['x']
    y_gt = labels

    # add output layer to the list of layers (this layer is used as an input for the first decoder layer)
    param_layers.append(y_gt.shape[0])

    with tf.device('/cpu:0'):

        # introduce noise to the inputs
        x_noisy = x + tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=param_n_sigma, dtype=tf.float32, name='n_0')

        '''Encoder with noise'''
        z_tilde_layers = []
        h_tilde_layers = []
        layer_inp = x_noisy
        for i, l in enumerate(param_layers):

            scope_name = 'L{:d}'.format(i)

            with tf.variable_scope(scope_name):
                z_tilde_l = tf.layers.dense(inputs = layer_inp, units = l, activation=None, use_bias=True, reuse = None)
                z_tilde_l += tf.random_normal(shape=tf.shape(z_tilde_l), mean=0.0, stddev=param_n_sigma, dtype=tf.float32, seed=None, name='n_{:d}'.format(i + 1))
                z_tilde_layers.append(z_tilde_l)
                if i < (len(param_layers)-1):
                    out = tf.contrib.layers.batch_norm(inputs=z_tilde_l, is_training=True, scale=True, activation_fn = tf.nn.relu)
                else:
                    out = tf.contrib.layers.batch_norm(inputs=z_tilde_l, is_training=True, scale=True, activation_fn=tf.nn.softmax)
                h_tilde_layers.append(out)
                layer_inp = out

        y_tilde = out

        # define supervised loss
        C_sv = tf.losses.softmax_cross_entropy(onehot_labels=y_gt, logits=y_tilde)

        #print([s.name for s in tf.global_variables()])

        '''Encoder w/o noise'''
        layer_inp = x
        mean_vectors = []
        std_vectors = []
        z_list = []
        for i, l in enumerate(param_layers):
            scope_name = 'L{:d}'.format(i)
            with tf.variable_scope(scope_name, reuse=True):
                z_pre = tf.layers.dense(inputs = layer_inp, units = l, activation=None, use_bias=True, reuse = True)

                mean_vectors.append(batchmean(z_pre))
                std_vectors.append(batchstd(z_pre))

                z = tf.contrib.layers.batch_norm(inputs=z_pre, is_training=True, scale=True)
                z_list.append(z)

                if i < (len(param_layers)-1):
                    h_l = tf.nn.relu(z)
                else:
                    h_l = tf.nn.softmax(z)

                layer_inp = h_l

        y_pred = h_l # use this op for prediction

        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)

        '''Decoder'''
        u_list = [y_tilde]
        C_uv = [] # unsupervised cost
        for i, l in enumerate(param_layers[::-1]):
            # denoising g(z,u)

            L_nr = str(len(param_layers) - i)

            if i==0:
                u_l = tf.contrib.layers.batch_norm(inputs=u_list[-1-i], is_training=True, scale=True)
                print('')
                z_hat_i = g(z_tilde_layers[-1-i], u_l, l)
            else:
                _ = tf.layers.dense(inputs=u_list[-1-i], units=l, name='V_'+L_nr)
                u_l = tf.contrib.layers.batch_norm(inputs=_, is_training=True, scale=True)
                z_hat_i = g(z_tilde_layers[-1-i], u_l, l)

            z_hat_i_BN = tf.divide(z_hat_i - mean_vectors[-1-i], std_vectors[-1-i])

            C_uv_l = param_importance[-1 - i] * tf.reduce_mean(tf.norm(z_hat_i_BN - z_list[-1 - i], axis=1), axis=0) / l
            C_uv.append(C_uv_l)

            u_list.append(z_hat_i_BN)

        C_uv_sum = 0

        # sum up unsupervised costs over all decoder layers
        for c in C_uv:
            C_uv_sum += c

        C_result = C_uv_sum + C_sv

        conf_matrix = tf.losses.confusion_matrix.confusion_matrix(labels=y_gt, predictions=y_pred)

        evaL_ops = {'C_sv': C_sv, 'C_uv': C_uv_sum, 'confusion_matrix': conf_matrix}

        train_iter = tf.train.GradientDescentOptimizer(learning_rate=param_lr).minimize(C_result)

    return tf.estimator.EstimatorSpec(predictions=y_pred, loss=C_result, train_op=train_iter, eval_metric_ops=evaL_ops)