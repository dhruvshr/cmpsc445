# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 2024

CMPSC 445 - M3 Assignment

@author: dhruv (dvs6026)
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# loading dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'label']
# dataset = pd.read_csv(url, names=names)

iris = load_iris()
iris_data = np.c_[iris.data, ['Iris-' + iris.target_names[i] for i in iris.target]]
iris_columns = iris.feature_names + ['class']
dataset = pd.DataFrame(data=iris_data, columns=iris_columns)

print(dataset.head())

"""
Partitioning Levels
A -> Training Set Size: 075 , Testing Set Size: 75
B -> Training Set Size: 100 , Testing Set Size: 50
C -> Training Set Size: 125 , Testing Set Size: 25
"""


# allocating features and targets
X = dataset.values[:,0:4].astype(float)
Y = dataset.values[:,4].astype(str)

ptn = 125  # dataset partition

# training set
X_train = np.array(X[:ptn])
Y_train = np.array(Y[:ptn])

# test set
X_test = np.array(X[ptn:])
Y_test = np.array(Y[ptn:])

# decision tree learning
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

# predictions
print("Predicting Labels...")
results = dtc.predict(X_test)

# determining accuracy
testset_size = len(Y_test)
correct_predictions = sum(results[i] == Y_test[i] for i in range(testset_size))
accuracy_score = correct_predictions / testset_size
accuracy_percentage = accuracy_score * 100

print()
print("Results:")
print(f"Testset Size      : {testset_size}")
print(f"Correct Preditions: {correct_predictions}")
print(f"Accuracy Percent  : {accuracy_percentage:.3f}%")