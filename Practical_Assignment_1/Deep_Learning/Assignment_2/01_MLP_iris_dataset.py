# MLP Program
# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
from sklearn.datasets import load_iris

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-Hot Encoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#  5. Build MLP Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create Model
model = Sequential()

# Hidden Layer 1
model.add(Dense(10, activation='relu', input_shape=(4,)))

# Hidden Layer 2
model.add(Dense(8, activation='relu'))

# Output Layer
model.add(Dense(3, activation='softmax'))

# 6. Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
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

#  9. Plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.title("Model Accuracy")
plt.show()

# 10. Prediction
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
print(predicted_classes)