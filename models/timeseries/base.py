import numpy as np
import tensorflow as tf


class TSLeNet5(tf.keras.Model):

    def __init__(self, num_classes, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        super(TSLeNet5, self).__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Conv1D(6, 5, activation='relu', padding='valid')
        self.pool1 = tf.keras.layers.MaxPool1D()
        self.conv2 = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.MaxPool1D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='relu')
        self.dense2 = tf.keras.layers.Dense(84, activation='relu')
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)

        return out


class TSFullyConvolutionalNetwork(tf.keras.Model):

    def __init__(self, num_classes, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        super(TSFullyConvolutionalNetwork, self).__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Conv1D(128, 8, padding='same',
                                            kernel_initializer='he_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2 = tf.keras.layers.Conv1D(256, 5, padding='same',
                                            kernel_initializer='he_uniform')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv3 = tf.keras.layers.Conv1D(128, 3, padding='same',
                                            kernel_initializer='he_uniform')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)

        self.gap = tf.keras.layers.GlobalAveragePooling1D()
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        x = self.gap(x)
        out = self.out(x)

        return out
