import pickle
import scipy
import numpy as np

# Load the data set
X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))

print("len X:")
print(len(X))
# print("X:")
# print(X)
print("len X[0]:")
print(len(X[0]))
print("X[0]:")
print(X[0])
print("len X[0][0]:")
print(len(X[0][0]))
print("X[0][0]:")
print(X[0][0])
print("len Y:")
print(len(Y))
print("Y:")
print(Y)