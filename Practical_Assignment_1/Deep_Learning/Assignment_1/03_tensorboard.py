import tensorflow as tf
import datetime
import os

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Correct log directory
log_dir = os.path.join(
    "logs",   # change from "log" → "logs"
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
os.makedirs(log_dir, exist_ok=True)

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

# Train
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback]
)

print("Logs saved in:", log_dir)
print("Run: tensorboard --logdir=logs")


'''
after running this program open terminal
in terminal:
paste --->  1. tensorboard --logdir=logs
            step 2 and 3 are for only run first time
            2. cd Practical_Assignment_1
            3. python -m tensorboard.main --logdir=logs
'''