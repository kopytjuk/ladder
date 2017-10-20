from keras.datasets import mnist
from ladder import model_fn
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from helpers import to_one_hot

if __name__ == '__main__':
    mnist_data = mnist.load_data()

    train = mnist_data[0]
    test = mnist_data[1] # one-hot

    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]

    X_train = X_train.reshape((-1, 28 * 28)).astype(np.float32)
    X_test = X_test.reshape((-1, 28 * 28)).astype(np.float32)

    y_train = to_one_hot(y_train, num=10)
    y_test = to_one_hot(y_test, num=10)

    assert isinstance(X_train, np.ndarray)

    # param_lr = params['learning_rate']
    # param_layers = params['layer_def']
    # param_importance = params['layer_importance']
    # param_device = params['device']  # cpu or gpu
    # param_n_sigma = params['n_sigma']  # noise value

    params = {'learning_rate': 1e-2,
              'layer_def': [20, 10],
              'layer_importance': [0.3333, 0.3333, 0.3333],
              'device': '/cpu:0',
              'n_sigma': 0.1}

    model = tf.estimator.Estimator(model_fn=model_fn, params=params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':X_train}, y=y_train, batch_size=128, num_epochs=10,
                                                        shuffle=True,num_threads=4)

    model.train(train_input_fn)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':X_test}, y=y_test, num_epochs=1,
                                                        shuffle=False, num_threads=4)

    ev = model.evaluate(test_input_fn)

    confusion_matrix = ev['confusion_matrix']

    print(confusion_matrix)
