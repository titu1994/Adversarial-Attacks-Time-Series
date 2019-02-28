import numpy as np
import tensorflow as tf


class TSFullyConnectedGATN(tf.keras.Model):

    def __init__(self, shape, units=200, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        super(TSFullyConnectedGATN, self).__init__(name=name, **kwargs)

        self.shape = shape
        self.units = units
        self.flat_dim = np.prod(shape)

        self.layer1 = tf.keras.layers.Dense(units, activation='relu', kernel_initializer='glorot_normal')
        self.layer2 = tf.keras.layers.Dense(self.flat_dim, activation='relu', kernel_initializer='glorot_normal')
        self.layer3 = tf.keras.layers.Dense(self.flat_dim, activation='linear', kernel_initializer='glorot_normal')
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input, grad, training=None, mask=None):
        input = self.flatten(input)
        grad = self.flatten(grad)

        ts = tf.concat([input, grad], axis=-1)
        x = self.layer1(ts)
        x = self.layer2(x)
        x = self.layer3(x + grad)

        output = (x + input)  # adding perturbation and norm

        output_shape = [output.shape[0]] + list(self.shape)
        output = tf.reshape(output, output_shape)

        return output


class TSConvGATN(tf.keras.Model):

    def __init__(self, shape, units=32, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        super(TSConvGATN, self).__init__(name=name, **kwargs)

        self.shape = shape
        self.units = units
        self.flat_dim = np.prod(shape)

        self.layer1 = tf.keras.layers.Conv1D(units, 3, activation='relu', padding='valid',
                                             kernel_initializer='glorot_normal')
        self.layer2 = tf.keras.layers.Conv1D(units, 3, activation='relu', padding='valid',
                                             kernel_initializer='glorot_normal')
        self.layer3 = tf.keras.layers.Dense(self.flat_dim, activation='relu', kernel_initializer='glorot_normal')
        self.layer4 = tf.keras.layers.Dense(self.flat_dim, activation='linear', kernel_initializer='glorot_normal')
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input, grad, training=None, mask=None):
        x1 = self.layer1(input)
        x2 = self.layer2(grad)

        input = self.flatten(input)
        grad = self.flatten(grad)
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        x = tf.concat([x1, x2], axis=-1)

        x = self.layer3(x)
        x = self.layer4(x + grad)  # x is perturbation

        output = (x + input)  # adding perturbation and norm

        output_shape = [output.shape[0]] + list(self.shape)
        output = tf.reshape(output, output_shape)

        return output
