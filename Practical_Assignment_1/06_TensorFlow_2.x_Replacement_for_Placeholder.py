#Use function arguments instead of placeholders.
import tensorflow as tf

@tf.function
def add(x, y):
    return x + y

print(add(10, 20))