import tensorflow as tf
import numpy as np

# Proper shapes
X = np.array([[1.],[2.],[3.],[4.],[5.]], dtype=np.float32)
y = np.array([[2.],[4.],[6.],[8.],[10.]], dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((X, y))

dataset = dataset.shuffle(buffer_size=5)
dataset = dataset.batch(2,drop_remainder=True)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Model (fixed input style)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')

model.fit(dataset,epochs=10)