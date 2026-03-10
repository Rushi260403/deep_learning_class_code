# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer


# 2. Load Dataset
cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

print(df.head())


# 3. Exploratory Data Analysis (EDA)

print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.pairplot(df[['mean radius', 'mean texture', 'mean area', 'target']], hue='target')
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

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# 5. Build ANN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(30,)))  # Hidden Layer 1
model.add(Dense(8, activation='relu'))                      # Hidden Layer 2
model.add(Dense(1, activation='sigmoid'))                   # Output Layer (Binary Classification)


# 6. Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# 7. Train Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=10,
    validation_split=0.2
)


# 8. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# 9. Plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend(['Train', 'Validation'])
plt.title('Model Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.show()


# 10. Prediction
predictions = model.predict(X_test)

predicted_classes = (predictions > 0.5).astype(int)

print("Predicted Classes:", predicted_classes[:10])
print("Actual Classes:", y_test.values[:10])