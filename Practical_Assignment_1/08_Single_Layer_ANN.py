# 2️. Single Layer ANN (Perceptron Model)
# Includes activation function (ReLU).
import tensorflow as tf

# Input
x = tf.constant([[2.0, 3.0]])

# Weights & Bias
W = tf.Variable([[1.0], [1.5]])
b = tf.Variable([0.5])

# Perceptron output
z = tf.matmul(x, W) + b
output = tf.nn.relu(z)

print("Z value:", z)
print("Activated Output:", output)
# 📌 Perceptron Equation
# output=f(Wx+b)output = f(Wx + b)output=f(Wx+b)