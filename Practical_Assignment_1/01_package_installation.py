import numpy
import matplotlib
import pandas as pd
import sklearn
import tensorflow as tf
print("Pandas version:", pd.__version__)
print("Scikit-learn imported successfully!")
print("TensorFlow version:", tf.__version__)
print("All libraries installed successfully!")

# This will make numpy and matplotlib turn white
data = numpy.array([1, 2, 3])
matplotlib.pyplot.plot(data)

# This will make sklearn turn white
print("Scikit-learn version:", sklearn.__version__)
