import tensorflow as tf

# Input data
x = tf.constant([[1.50, 2.0]])   # shape (1, 2)

# Weights and bias
W = tf.Variable([[1.5], [0.8]]) # shape (2, 1)
b = tf.Variable([0.3])          # shape (1,)

# ANN computation
y = tf.matmul(x, W) + b

print("Output:", y)
