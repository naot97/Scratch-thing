import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each ROW of Z is a set of scores.
    """
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A
def crossentropy_loss(Yhat, y):
    """
    Yhat: a numpy array of shape (Npoints, nClasses) -- predicted output
    y: a numpy array of shape (Npoints) -- ground truth.
    NOTE: We don’t need to use the one-hot vector here since most of elements
    are zeros. When programming in numpy, in each row of Yhat, we need to access
    to the corresponding index only.
    """
    id0 = range(Yhat.shape[0])
    return -np.mean(np.log(Yhat[id0, y]))
def mlp_init(d0, d1, d2):
    """
    Initialize W1, b1, W2, b2
    d0: dimension of input data
    d1: number of hidden unit
    d2: number of output unit = number of classes
    """
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros(d1)
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros(d2)
    return (W1, b1, W2, b2)
def mlp_predict(X, W1, b1, W2, b2):
    """
    Suppose that the network has been trained, predict class of new points.
    X: data matrix, each ROW is one data point.
    W1, b1, W2, b2: learned weight matrices and biases
    """
    Z1 = X.dot(W1) + b1 # shape (N, d1)
    A1 = np.maximum(Z1, 0) # shape (N, d1)
    Z2 = A1.dot(W2) + b2 # shape (N, d2)
    return np.argmax(Z2, axis=1)

def mlp_fit(X, y, W1, b1, W2, b2, eta):
    loss_hist = []
    for i in range(100): # number of epoches
    # feedforward
        Z1 = X.dot(W1) + b1 # shape (N, d1)
        A1 = np.maximum(Z1, 0) # shape (N, d1)
        Z2 = A1.dot(W2) + b2 # shape (N, d2)
        Yhat = softmax_stable(Z2) # shape (N, d2) = A2
        # back propagation
        id0 = range(Yhat.shape[0])
        Yhat[id0, y] -=1
        E2 = Yhat/N # shape (N, d2)
        dW2 = np.dot(A1.T, E2) # shape (d1, d2)
        db2 = np.sum(E2, axis = 0) # shape (d2,)
        E1 = np.dot(E2, W2.T) # shape (N, d1)
        E1[Z1 <= 0] = 0 # gradient of ReLU, shape (N, d1)
        dW1 = np.dot(X.T, E1) # shape (d0, d1)
        db1 = np.sum(E1, axis = 0) # shape (d1,)
        # Gradient Descent update
        W1 += -eta*dW1
        b1 += -eta*db1
        W2 += -eta*dW2
        b2 += -eta*db2
    return (W1, b1, W2, b2)

def mlp_test(W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1 # shape (N, d1)
    A1 = np.maximum(Z1, 0) # shape (N, d1)
    Z2 = A1.dot(W2) + b2 # shape (N, d2)
    Yhat = softmax_stable(Z2)
    id0 = np.argmax(Yhat, axis=1)
    print(list(y == id0).count(True)/N )


data = pd.read_csv('train.csv').values
X = data[0:100,1:]
y = data[:100,0]
N = 1000
d0 = X[0].size # data dimension
d1 = h = 100 # number of hidden units
d2 = C = 10 # number of classes
eta = 20 # learning rate
(W1, b1, W2, b2) = mlp_init(d0, d1, d2)
(W1, b1, W2, b2) = mlp_fit(X, y, W1, b1, W2, b2, eta)
mlp_test(W1,b1,W2,b2);
''''y_pred = mlp_predict(X, W1, b1, W2, b2)
acc = 100*np.mean(y_pred == y)
print('training accuracy: %.2f %%' % acc)'''