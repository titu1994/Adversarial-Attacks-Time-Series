import numpy as np
import tensorflow as tf


class TSDistilledFullyConnected(tf.keras.Model):

    def __init__(self, num_classes, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        super(TSDistilledFullyConnected, self).__init__(name=name, **kwargs)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.out = tf.keras.layers.Dense(num_classes, activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)

        if training is False:
            out = tf.nn.softmax(out, axis=-1)

        return out


class TSDistilledLeNet5(tf.keras.Model):

    def __init__(self, num_classes, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        super(TSDistilledLeNet5, self).__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Conv1D(6, 5, activation='relu', padding='valid')
        self.pool1 = tf.keras.layers.MaxPool1D()
        self.conv2 = tf.keras.layers.Conv1D(16, 5, activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.MaxPool1D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='relu')
        self.dense2 = tf.keras.layers.Dense(84, activation='relu')
        self.out = tf.keras.layers.Dense(num_classes, activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)

        if training is False:
            out = tf.nn.softmax(out, axis=-1)

        return out


class _ConvBNRelu(tf.keras.Model):

    def __init__(self, filters, kernel, stride, activation=tf.nn.relu):
        super(_ConvBNRelu, self).__init__()

        self.conv = tf.keras.layers.Conv1D(filters, kernel, stride, padding='same',
                                           kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = activation

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)

        return x


class TSDistilledMiniVGG(tf.keras.Model):

    def __init__(self, num_classes, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        super(TSDistilledMiniVGG, self).__init__(name=name, **kwargs)

        self.gap = tf.keras.layers.GlobalAveragePooling1D()

        self.layer1_1 = _ConvBNRelu(32, kernel=8, stride=1)
        self.layer1_2 = _ConvBNRelu(64, kernel=8, stride=1)
        self.layer1_pool = tf.keras.layers.MaxPool1D()

        self.layer2_1 = _ConvBNRelu(64, kernel=5, stride=1)
        self.layer2_2 = _ConvBNRelu(128, kernel=5, stride=1)
        self.layer2_3 = _ConvBNRelu(128, kernel=5, stride=1)
        self.layer2_pool = tf.keras.layers.MaxPool1D()

        self.layer3_1 = _ConvBNRelu(128, kernel=3, stride=1)
        self.layer3_2 = _ConvBNRelu(256, kernel=3, stride=1)
        self.layer3_3 = _ConvBNRelu(256, kernel=3, stride=1)

        self.out = tf.keras.layers.Dense(num_classes, activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.layer1_1(inputs, training=training)
        x = self.layer1_2(x, training=training)
        x = self.layer1_pool(x)

        x = self.layer2_1(x, training=training)
        x = self.layer2_2(x, training=training)
        x = self.layer2_3(x, training=training)
        x = self.layer2_pool(x)

        x = self.layer3_1(x, training=training)
        x = self.layer3_2(x, training=training)
        x = self.layer3_3(x, training=training)

        x = self.gap(x)
        out = self.out(x)

        if training is False:
            out = tf.nn.softmax(out, axis=-1)

        return out
