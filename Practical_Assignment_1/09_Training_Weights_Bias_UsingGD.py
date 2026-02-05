# 3️.Training Weights & Bias Using Gradient Descent
# Simple ANN learning example.
import tensorflow as tf

# Input and target
x = tf.constant([[1.0], [2.0], [3.0]])
y_true = tf.constant([[2.0], [4.0], [6.0]])

# Initialize weights & bias
W = tf.Variable([[0.5]])
b = tf.Variable([0.0])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Training loop
for epoch in range(10):
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(x, W) + b
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    grads = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))

    print(f"Epoch {epoch}: Loss = {loss.numpy()}")