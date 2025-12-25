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

# filter: Keeping only class 1 (versicolor) and class 2 (virginica) and keep only the last 2 features (petal length, petal width)
X, y = X[y != 0][:, 2:], y[y != 0]

# converting the labels to -1 and 1
y = np.where(y == 1, -1, 1)

# normalization
X = (X - X.mean(axis=0)) / X.std(axis=0) # axis = 0 : it collapses the rows and runs downwards

# cmaes loop
w = np.zeros(2) # we again set the starting values 
b = 0 # bias

best_global_loss = float("inf") 
best_global_x = None # vector to log the best weights (along with the bias)

def get_accuracy(w, b):
  preds = np.where((np.dot(X, w)+b) >= 0, 1 , -1)
  return np.mean (preds == y)

# initialization of cma-es optimizer
dim = 3
mean = np.zeros(dim)
sigma = 1.0

optimizer = CMA(mean=mean, sigma = sigma)

n_generations = 200

for gen in range(n_generations):
  solutions = []

  for i in range(optimizer.population_size):
    x = optimizer.ask()
    w_candidate = x[:2]
    b_candidate = x[2]
    accuracy = get_accuracy(w_candidate, b_candidate)
    loss = -accuracy
    solutions.append((x, loss))

  optimizer.tell(solutions)

  gen_best_x, gen_best_loss = min(solutions, key=lambda s: s[1]) # we are comparing and ranking the 2nd element


  if gen_best_loss < best_global_loss:
    best_global_loss = gen_best_loss
    best_global_x = gen_best_x.copy()

  if gen % 20 == 0:
    print(f"Gen {gen:3d} | accuracy = {-best_global_loss:.4f}")

w = best_global_x[:2]
b = best_global_x[2]

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

plt.plot(x_vals, y_vals, 'g-', linewidth=3, label='cma-es boundary')

plt.xlabel("petal length (normalized)")
plt.ylabel("petal width (normalized)")
plt.legend()
plt.grid(True, alpha=0.5) # 50% opacity in the grid
plt.show()