import math,copy
import numpy as np
import matplotlib.pyplot as plt
from pip._internal.req.req_file import preprocess


# ////////////////////////////////////////
# compute model function without victorization
# def compute_model_output(x,w,b):
#     m=x.shape[0]
#     fx=0
#     for i in range(m):
#         fx_i=x[i]*w[i]
#         fx=fx+fx_i
#     fx=fx+b
#     return fx
# //////////////////////////////////////
# comput model function with victorization
def compute_model_output_v(x,w,b):
    # x : ndarray(n,)
    # w : ndarray(n,)
    # b : scalar
    fx=np.dot(x,w)+b
    return fx
# ////////////////////////////////////////////
# compute cost function
def compute_cost(X,y,w,b):
    # X : ndarray(m,n)
    # w : ndarray(n,)
    # y : ndarray(n,)
    # b : scalar
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        fx_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (fx_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost
# //////////////////////////////////
# compute the derivative of w and b
def compute_derivative(X,y,w,b):
    # X : ndarray(m,n)
    # w : ndarray(n,)
    # y : ndarray(n,)
    # b : scalar
    # dj_dw : nsarray(n,)
    # dj_db : scalar
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        temp=(np.dot(X[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+temp*X[i,j]
        dj_db=dj_db+temp
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db
# ///////////////////////////////////////////
# compute gradient descent to optimize variables
def gradient_desent(X,y,w_in,b_in,alpha,iterations,compute_cost,compute_derivative):
    # save cost function values to plot them later
    J_history=[]
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b=b_in
    for i in range(iterations):
        dj_dw,dj_db=compute_derivative(X,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history  # return final w,b and J history for graphing
# //////////////////////////////////////////////////////////////////////////////////////////
# plot cost ax1 from 0 to 100 and ax2 from 100 till end
def plot_cost(j_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(j_history[:100])
    ax2.plot(100 + np.arange(len(j_history[100:])), j_history[100:])
    ax1.set_title("Cost vs. iteration(start)");
    ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel('Cost');
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step');
    ax2.set_xlabel('iteration step')
    plt.show()
# /////////////////////////////////////////////////////////////////////////////////////////
# zscore normalization
def zscore_norm(X):
    # X : ndarray(m,n)
    # mu : ndarray(n,)
    # sigma :ndarray(n,)
    mu=np.mean(X)
    sigma=np.std(X)
    X_norm=(X-mu)/sigma
    return  X_norm,mu,sigma
# ////////////////////////////////////////////////////////////////////////////////////////////
# plot cost 2d
def plot2d(x, y, mode):
    plt.title("DATA")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x, y, mode)
    plt.show()
    return
# *********************************************************************************************
# X_train = np.array([
# [2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
# y_train = np.array([460, 232, 178])

# import our data from a csv file
data= np.genfromtxt("data2.csv", delimiter=',')
print("data dimensions= ",data.shape)
numOfFeatures = np.size(data, 1) - 1
print(numOfFeatures)

X_train = data[:, 0:numOfFeatures]
y_train = data[:, numOfFeatures]

m,n = X_train.shape


# print(X_train)
# use mu and sigma to normalize new training examples
X_norm,mu,sigma=zscore_norm(X_train)
# print(X_norm)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741])

initial_w=np.zeros_like(w_init)
initial_b=0.
alpha=0.03e-1
iterations=1000
final_w,final_b,J_history=gradient_desent(X_norm,y_train,initial_w,initial_b,alpha,iterations,compute_cost,compute_derivative)
print(f"b,w found by gradient descent: {final_b:0.2f},{final_w} ")
for i in range(m):

    print(f"prediction: {np.dot(X_norm[i], final_w) + final_b:0.2f}, target value: {y_train[i]}")

plot_cost(J_history)

# plot cost function to the number of iterations
x_axis = np.mat(range(0, iterations)).reshape(( iterations,1))
plot2d(x_axis, J_history, ".b")
#
# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features //// polynomial regression
X =x**2

X = X.reshape(-1, 1)
#X should be a 2-D Matrix
X = X / X.mean(axis=0, keepdims=True)

print(X)

b=0
w=np.zeros_like(X.shape[1])
iter=1000
a=1e-1
model_w,model_b,j_values = gradient_desent(X, y,w,b, a, iter,compute_cost,compute_derivative)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()