import tensorflow as tf
from helpers import batchmean, batchstd, g


def model_fn(features, labels, mode, params):
    """
    :param features: dict with values as input tensors
    :param labels: output tensor
    :param mode: TRAIN, EVAL or PREDICT
    :param params: dict with model parameters
    :return: EstimatorSpec
    """

    global out
    param_lr = params['learning_rate']
    param_layers = params['layer_def']
    param_importance = params['layer_importance']
    param_device = params['device'] # cpu or gpu
    param_n_sigma = params['n_sigma'] # noise value
    param_num_class = params['num_class']

    x = features['x']
    y_gt = labels

    # add output layer to the list of layers (this layer is used as an input for the first decoder layer)
    param_layers.append(param_num_class)

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
                    out = tf.layers.batch_normalization(inputs=z_tilde_l, training=True, reuse=None)
                    out = tf.nn.relu(out)
                else:
                    out = tf.layers.batch_normalization(inputs=z_tilde_l, training=True, reuse=None)
                    out = tf.nn.softmax(out)

                h_tilde_layers.append(out)
                layer_inp = out

        # final layer output used as vector for supervised cost
        y_tilde = out

        if mode == tf.estimator.ModeKeys.PREDICT:
            bn_training = False
        else:
            bn_training = True

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

                #z = tf.contrib.layers.batch_norm(inputs=z_pre, is_training=True, scale=True)
                z = tf.layers.batch_normalization(inputs=z_pre, training=bn_training, reuse=True)

                z_list.append(z)

                if i < (len(param_layers)-1):
                    h_l = tf.nn.relu(z)
                else:
                    h_l = tf.nn.softmax(z)

                layer_inp = h_l

        y_pred = h_l # use this op for prediction after training

        if mode == tf.estimator.ModeKeys.PREDICT:
            # calc index of the class with maximum softmax output
            winner_class = tf.argmax(y_pred, dimension=1)
            return tf.estimator.EstimatorSpec(mode=mode, predictions={'y': winner_class})

        '''Decoder'''
        u_list = [y_tilde]
        C_uv_list = [] # unsupervised cost
        for i, l in enumerate(param_layers[::-1]):

            L_nr = str(len(param_layers) - i)

            if i==0:
                u_l = tf.contrib.layers.batch_norm(inputs=u_list[-1-i], is_training=True, scale=True)
                z_hat_i = g(z_tilde_layers[-1-i], u_l, l)
            else:
                _ = tf.layers.dense(inputs=u_list[-1-i], units=l, name='V_'+L_nr)
                u_l = tf.contrib.layers.batch_norm(inputs=_, is_training=True, scale=True)
                z_hat_i = g(z_tilde_layers[-1-i], u_l, l)

            z_hat_i_BN = tf.divide(z_hat_i - mean_vectors[-1-i], std_vectors[-1-i])

            C_uv_l = param_importance[-1 - i] * tf.reduce_mean(tf.norm(z_hat_i_BN - z_list[-1 - i], axis=1), axis=0) / l
            C_uv_list.append(C_uv_l)

            u_list.append(z_hat_i_BN)

        # define supervised loss
        C_sv = tf.losses.softmax_cross_entropy(onehot_labels=y_gt, logits=y_tilde)

        # sum up unsupervised costs over all decoder layers
        C_uv = 0
        for c in C_uv_list:
            C_uv += c

        # overall cost
        C_all = C_uv + C_sv

        #conf_matrix = tf.losses.confusion_matrix.confusion_matrix(labels=y_gt, predictions=y_pred)

        #evaL_ops = {'C_sv': C_sv, 'C_uv': C_uv_sum, 'confusion_matrix': conf_matrix}
        #conf_matrix = tf.metrics.confusion_matrix.confusion_matrix(labels=y_gt,predictions=y_pred)
        acc = tf.metrics.accuracy(labels=tf.argmax(y_gt),predictions=tf.argmax(y_pred))
        eval_ops = {'accuracy': acc}
        #eval_ops = {'confusion_matrix': conf_matrix}
        #eval_ops = {'C_sv': C_sv, 'C_uv': C_uv_sum}

        train_iter = tf.train.AdamOptimizer(learning_rate=param_lr).minimize(C_all, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(predictions=y_pred, loss=C_all, train_op=train_iter, eval_metric_ops=eval_ops, mode=mode)