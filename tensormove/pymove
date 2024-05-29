import tensorflow as tf
from tensorflow.keras import layers, models

class TensorFlowModel(tf.keras.Model):
    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu', input_shape=(784,))
        self.fc2 = layers.Dense(10)

    def call(self, inputs):
        x = self.fc1(inputs)
        return self.fc2(x)

tensorflow_model = TensorFlowModel()
