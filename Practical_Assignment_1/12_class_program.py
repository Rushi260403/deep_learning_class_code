import tensorflow as tf

# Disable eager execution (VERY IMPORTANT)
tf.compat.v1.disable_eager_execution()

x1 = tf.compat.v1.placeholder(tf.float32, name="x1")
x2 = tf.compat.v1.placeholder(tf.float32, name="x2")

multiply = tf.multiply(x1, x2, name="multiply")

with tf.compat.v1.Session() as sess:
    result = sess.run(
        multiply,
        feed_dict={x1: [1, 2, 3], x2: [4, 5, 6]}
    )
    print(result)
