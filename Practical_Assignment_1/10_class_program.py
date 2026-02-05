import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = a * b
sess=tf.Session()
File_Writer = tf.summary.FileWriter("./logs", sess.graph)
print(sess.run(c,feed_dict={a: [1, 2, 3], b: [4, 5, 6]}))