import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
(training_data, training_labels), (test_data, test_labels) = mnist.load_data()
print(type(training_data[0]))
