# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    e = y - tx.dot(w)
    L = 1/(2*np.shape(tx)[0]) * e.dot(e.T)
    return L



""" Grid Search"""

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search
#       here when it is done.
# ***************************************************
def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    for i in range (0, len(w0)):
        for j in range(0, len(w1)):
            losses[i,j] = compute_loss(y, tx, np.array([w0[i], w1[j]]).T)
    return losses


"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and loss
    # ***************************************************
    e = y - tx.dot(w)
    gradient = -1/(np.shape(tx)[0]) * tx.T.dot(e)
    return gradient    


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    e = y - tx.dot(w)
    gradient = -1/(np.shape(tx)[0]) * tx.T.dot(e)
    return gradient    

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        y_batch, tx_batch = [batch for batch in batch_iter(y,tx, batch_size)][0]
        gradient = compute_gradient(y_batch,tx_batch,w)
        loss = compute_loss(y,tx,w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, ws

"""Least Squares"""

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w_opt = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    #w_opt = np.linalg.solve(tx,y)
    e = y - tx.dot(w_opt)
    mse = 1/(2*np.shape(tx)[0]) * e.dot(e.T)
    
    return mse, w_opt

"""Build polynomial basis"""

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    _x = np.ones((np.shape(x)[0], 1 + np.shape(x)[1]*degree))
    for i in range(0, degree):
        for k in range(0, np.shape(x)[1]):
            _id = 1 + i*np.shape(x)[1] + k
            _x[:, _id] = x[:, k]**(i+1)
    return _x

"""Ridge Regression"""

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************

    L = np.eye(tx.shape[1])*lambda_*2*len(y)
    weights= np.linalg.inv(tx.T.dot(tx) + L.T.dot(L)).dot(tx.T).dot(y)
    loss = compute_loss(y,tx,weights)
    
    return weights, loss

"""Random split data"""

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # split the data based on the given ratio: 
    # ***************************************************
    temp = np.random.rand(x.shape[0])
    ind = np.argsort(temp)
    
    lim = int(np.floor(ratio*x.shape[0]))
    x_train = np.zeros(lim)
    y_train = np.zeros(lim)
    x_test = np.zeros(x.shape[0]-lim)
    y_test = np.zeros(x.shape[0]-lim)

    for i in range(0,lim):
        x_train[i] = x[ind[i]]
        y_train[i] = y[ind[i]]
        
    for j in range(lim,x.shape[0]):
        x_test[j-lim] = x[ind[j]]
        y_test[j-lim] = y[ind[j]]   
        
    return x_train, y_train, x_test, y_test