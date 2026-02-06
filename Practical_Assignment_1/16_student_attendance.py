# Step 1: Import Required Libraries
import tensorflow as tf
import pandas as pd
import numpy as np


#Step 2: Load Student Data (CSV)
data = pd.DataFrame({
    "attendance": [75, 40, 85, 60, 30],
    "internal_marks": [65, 30, 78, 55, 25],
    "assignment_marks": [70, 35, 80, 58, 20],
    "study_hours": [3, 1, 4, 2, 1],
    "result": [1, 0, 1, 1, 0]
})

X = data[['attendance', 'internal_marks', 'assignment_marks', 'study_hours']]
y = data['result']   # 1 = Pass, 0 = Fail


#Step 3: Build ANN Model (TensorFlow)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


#Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


#Step 5: Train the ANN
model.fit(X, y, epochs=50, batch_size=5)


#Step 6: Predict Pass / Fail for New Student
new_student = np.array([[72, 60, 65, 2]])

prediction = model.predict(new_student)

if prediction >= 0.5:
    print("Student Result: PASS")
else:
    print("Student Result: FAIL")
