import numpy as np
import tensorflow as tf

x_input = np.random.sample((1,2))
print(x_input)

x = tf.placeholder(tf.float32, shape=[1,2], name='x')
dataset = tf.data.Dataset.from_tensor_slices(x)

iterator = dataset.make_initializable_iterator()
get_next = iterator.get_next()
with tf.session() as sess:
    sess.run(iterator.Initializer, feed_dict={x,x_input})
    print(sess.run(get_next))