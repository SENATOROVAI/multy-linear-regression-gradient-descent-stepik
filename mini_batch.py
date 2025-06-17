def mini_batch_gradient_descent(X, y, batch_size=2, learning_rate=0.01, epochs=2):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    loss_history = []
    for epoch in range(epochs):
        # Shuffle data to ensure randomness
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute predictions
            predictions = np.dot(X_batch, weights) + bias
            
            # Compute error
            error = predictions - y_batch
            
            # Compute gradients
            weights_gradient = np.dot(X_batch.T, error) / batch_size
            bias_gradient = np.sum(error) / batch_size
            
            # Update parameters
            weights -= learning_rate * weights_gradient
            bias -= learning_rate * bias_gradient
        
        # Compute loss (Mean Squared Error)
        loss = np.mean(error ** 2)
        loss_history.append(loss)
    
    return weights, bias, loss_history



X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])



y = np.array([5,8,11,14,17])

weights, bias, loss_history = mini_batch_gradient_descent(X, y)
