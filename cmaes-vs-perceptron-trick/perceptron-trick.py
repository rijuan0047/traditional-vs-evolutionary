import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from cmaes import CMA

# loading the iris dataset
iris = load_iris() # it returns the iris dictionary
# print(iris.keys()) # this shows the available filed names within the dictionary iris
# print(iris.DESCR) # iris.DESCR gives the description of the orgnization of the data
# print(iris.target) # prints the output classes
df = pd.DataFrame(iris.data, columns = iris.feature_names)

X = iris.data # X is the feature matrix
y = iris.target # Y is the vector representing output classes

# filter: Keeping only class 1 (versicolor) and class 2 (virginica) and keeping only the last 2 features (petal length, petal width)
X, y = X[y != 0][:, 2:], y[y != 0]

# converting the labels to -1 and 1
y = np.where(y == 1, -1, 1)

# normalization
X = (X - X.mean(axis=0)) / X.std(axis=0) # axis = 0 : it collapses the rows and runs downwards

# the plot for visualization
# plt.figure(figsize=(10, 6))
# plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='versicolor (-1)')
# plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='virginica (+1)')

# plt.title("iris dataset: versicolor vs. virginica")
# plt.xlabel("petal length (normalized)")
# plt.ylabel("petal width (normalized)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

print("start of perceptron trick training")
np.random.seed(0)
w = np.zeros(2) # we have two weights for two features
b = 0 # bias
learning_rate = 0.01
epochs = 100

# the perceptron trick loop

perceptron_trick_history = [] # storing the lines (w, b)

for epoch in range (epochs):

    for i, x_i in enumerate(X): # iterating row by row
      # prediction
      linear_output = np.dot(x_i, w) + b
      y_pred = 1 if linear_output >= 0 else -1

      # the perceptron trick
      if y_pred != y[i]:
        update = learning_rate * (y[i] - y_pred)
        w += update*x_i
        b += update
      
      perceptron_trick_history.append((w.copy(), b))

# visualizing 
plt.figure(figsize=(10, 6))
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='versicolor (-1)')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='virginica (+1)')


# comparing real output vs predicted output
linear_output = np.dot(X, w) + b
predictions = np.where(linear_output >= 0, 1, -1)

# the accuracy in perceptron trick
accuracy = np.mean(predictions == y)
print(accuracy)

# drawing the decision boundary
x_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]) # x points to plot the line
y_vals = -(w[0] * x_vals + b) / w[1] # equation: w1*x + w2*y + b = 0  =>  y = -(w1*x + b) / w2

plt.plot(x_vals, y_vals, 'g-', linewidth=3, label='perceptron trick boundary')

plt.xlabel("petal length (normalized)")
plt.ylabel("petal width (normalized)")
plt.legend()
plt.grid(True, alpha=0.5) # 50% opacity in the grid
plt.show()