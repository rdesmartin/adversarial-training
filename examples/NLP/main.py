from ...src import train
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np


def get_model(input_size):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(128, activation='relu', kernel_initializer=initializer, name='dense_1'),
            layers.Dense(2, activation='linear', kernel_initializer=initializer, name='output_layer')
        ])
    print(model.summary())
    return model


if __name__ == '__main__':
    input_size = 30
    epochs = 50
    batch_size = 64
    pgd_steps = 5
    alpha = 1
    beta = 1
    eps_multiplier = 1000
    from_logits = True

    model = get_model(input_size)

    X_train = np.load(f'examples/NLP/data/X_train.npy')
    X_test = np.load(f'examples/NLP/data/X_test.npy')
    y_train = np.load(f'examples/NLP/data/y_train.npy')
    y_test = np.load(f'examples/NLP/data/y_test.npy')

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    hyperrectangles = np.load(f'examples/NLP/hyperrectangles/hyperrectangles.npy')
    hyperrectangles_labels = np.full(len(hyperrectangles), 0)

    model = train(model, train_dataset, test_dataset, epochs, batch_size, pgd_steps, hyperrectangles, hyperrectangles_labels, alpha, beta, eps_multiplier, from_logits)

    # Test the model
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    for x_batch_test, y_batch_test in test_dataset:
        test_outputs = model(x_batch_test, training=False)
        test_acc_metric.update_state(y_batch_test, test_outputs)

    print(f'{float(test_acc_metric.result()):.4f}')