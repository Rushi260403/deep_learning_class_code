#Variables are used to store trainable parameters like weights and biases.
#✔ TensorFlow 2.x
import tensorflow as tf

# Create a variable
x = tf.Variable(5)
y = tf.Variable([1.0, 2.0, 3.0])

print(x)
print(y)

#🔁 Updating a Variable
x.assign(10)
print("Updated value:", x)