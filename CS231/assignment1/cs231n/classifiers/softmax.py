import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = np.dot(X[i, :], W)
    scores -= np.max(scores)
    loss -= scores[y[i]]
    
    p = 0.0
    for s in scores:
      p += np.exp(s)
    
    loss += np.log(p.sum())
    for j in xrange(num_classes):
      dW[:, j] += X[i, :].T * np.exp(scores[j])/p.sum()
      if y[i] == j:
          dW[:, j] -= X[i, :].T
  print(X[i, :].T.shape)
  
  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  scores = np.dot(X, W)
  scores -= np.max(scores, axis=1)[:, np.newaxis]
  score_y = scores[np.arange(num_train), y]

  exp_sum = np.sum(np.exp(scores), axis=1)[:, np.newaxis]
  loss = -score_y.sum() + np.log(exp_sum).sum()
    
  zz = np.zeros(scores.shape)
  zz[np.arange(num_train), y] = 1
  pp = np.exp(scores)/exp_sum
  dW = np.dot(X.T, pp) - np.dot(X.T, zz)
  
  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

