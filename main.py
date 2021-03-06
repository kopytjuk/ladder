from keras.datasets import mnist
from ladder import model_fn
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from helpers import to_one_hot, uniform_layer_importance

tf.logging.set_verbosity(tf.logging.INFO)

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

    N = 30000

    idx = np.random.randint(0, X_train.shape[0], N)

    X_train = X_train[idx,:]
    y_train = y_train[idx,:]

    assert isinstance(X_train, np.ndarray)

    params = {'learning_rate': 1e-2,
              'layer_def': [20, 10],
              'layer_importance': [0.3333, 0.3333, 0.3333],
              'device': '/cpu:0',
              'n_sigma': 0.1,
              'num_class': 10}

    with tf.device('/cpu:0'):
        model = tf.estimator.Estimator(model_fn=model_fn, params=params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':X_train}, y=y_train, batch_size=1000, num_epochs=20,
                                                        shuffle=True, num_threads=1)

    model.train(train_input_fn)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':X_test}, y=y_test, num_epochs=1,
                                                        shuffle=False, num_threads=1)

    ev = model.evaluate(test_input_fn)

    pr = model.predict(test_input_fn)

    #confusion_matrix = ev['confusion_matrix']
    #print(confusion_matrix)
    print(ev['accuracy'])
    print(y_test[:5])

    for i in range(5):
        print(next(pr)['y']+1)
