import numpy as np
from numpy.random import normal
from numpy.linalg import norm
from util import project_onto_l1_ball, multiconv
from sklearn.preprocessing import normalize

def learn_constrained_sparse_dictionary(X, n, K, \
                                        lambda_D_smooth = 0.01, \
                                        lambda_R_sparse = 0.01, \
                                        gamma_0 = 0.01, \
                                        num_iterations = 200, \
                                        zero_edge = False):
  """
  Decomposes a (column vector) signal X into the sum over j = 1:k of
  convolutions of sparse `position' vectors R_j with short, smooth `feature' vectors D_j
  
  Required Inputs:
    X (real, length=N):     signal to decompose into R and D (such that sum_j R_j * D_j approximates X)
    n (integer scalar > 0): desired length of features
    K (integer scalar > 0): desired number of features
  
  Optional Inputs:
    lambda_D_smooth (real scalar >= 0,    Default: 0.01) weight of smoothness penalty on D
    lambda_R_sparse (real scalar > 0,     Default: 0.01) weight of sparsity penalty on R
    gamma_0         (real scalar >= 0,    Default: 0.01) initial gradient descent step size
    num_iterations  (integer scalar >= 0, Default: 100)  total number of gradient descent iterations
  
  Outputs:
    R (non-negative, real, sparse, (N - n + 1) X K): matrix whose jth column encodes positions at which feature j occurs
    D (smooth, real, n X K): matrix whose j^th column encodes the j^th feature
    opt (non-negative, real, length=num_iterations): history of objective function
  """

  N = X.shape[0]
  
  # Initialize each D_j as a uniformly random unit vector (with 0's at each end)
  D = normal(0, 1, (n, K))
  if zero_edge:
    D[0, :] = 0
    D[n - 1, :] = 0
  normalize(D, axis = 0, copy = False)

  # Initialize each R_j as a non-negative IID gaussian vector
  R_shape = (N - n + 1, K)
  R = np.maximum(normal(0, norm(X)/K, R_shape), 0)

  # TODO: implement optimization procedure

  return R, D#, reconstruction_error
