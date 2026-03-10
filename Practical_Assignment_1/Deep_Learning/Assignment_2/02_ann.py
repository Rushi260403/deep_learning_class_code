# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.datasets import load_iris


# 2. Load Dataset
data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())


# 3. Exploratory Data Analysis (EDA)

print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.pairplot(df, hue='target')
plt.show()


# 4. Dataset Preparation
X = df.drop('target', axis=1)
y = df['target']


# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Build ANN Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),  # Hidden Layer
    tf.keras.layers.Dense(3, activation='softmax')  # Output Layer
])


# 6. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 7. Train Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2
)


# 8. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# 9. Plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend(['Train', 'Validation'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.show()


# 10. Prediction
predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)

print("Predicted Classes:", predicted_classes[:10])
print("Actual Classes:", y_test.values[:10])