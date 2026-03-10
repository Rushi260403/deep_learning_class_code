#❌ Not used in TensorFlow 2.x
#✔ Used when data is fed at runtime
#✔ TensorFlow 1.x Style Code
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Create placeholders
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Operation
z = x + y

with tf.Session() as sess:
    result = sess.run(z, feed_dict={x: 10.5, y: 20.3})
    print("Result:", result)

#Placeholder with Shape (TF 1.x)
x = tf.placeholder(tf.float32, shape=[None, 3])
#None → any number of rows
#3 → exactly 3 columns