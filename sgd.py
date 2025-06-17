import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def sgd(X, y, learning_rate=0.01, epochs=2, batch_size=1):
    m = len(X)  
    theta = np.random.randn(3, 1) 
    
    X_bias = np.c_[np.ones((m, 1)), X]

    cost_history = []  

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X_bias[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            gradients = 2 / batch_size * X_batch.T.dot(X_batch.dot(theta) - y_batch)
            theta -= learning_rate * gradients

        predictions = X_bias.dot(theta)
        cost = np.mean((predictions - y) ** 2)
        cost_history.append(cost)

    return theta, cost_history
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])
# y = X[:, -1]**2  # y = 3X + noise

y = np.array([5,8,11,14,17])

theta_final, cost_history = sgd(X, y, learning_rate=0.1, epochs=2, batch_size=1)
