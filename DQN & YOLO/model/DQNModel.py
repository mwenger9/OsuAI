import tensorflow as tf

class DQNModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation="relu")
        self.fc2 = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x