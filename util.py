import numpy as np

def project_onto_l1_ball(v, b):
# project_onto_l1_ball Projects point onto L1 ball of specified radius.
#
# project_onto_l1_ball(v, b) returns the vector w which is the solution
#   to the following constrained minimization problem:
#
#     min   ||w - v||_2
#     s.t.  ||w||_1 <= b.
#
#   That is, performs Euclidean projection of v to the 1-norm ball of radius b.
#
# Author: Shashank Singh (sss1@andrew.cmu.edu)
# Adapted the from Matlab code of John Duchi (jduchi@cs.berkeley.edu)

  if (b < 0):
    raise ValueError('Radius of L1 ball is negative: %2.3f\n', b)

  if sum(abs(v)) < b: # If v already lies in the L1 ball
    return v

  v = v.astype(float) # otherwise, there can be issues with integer division
  u = np.sort(abs(v))[::-1] # sort entries of v decreasing by magnitude
  sv = np.cumsum(u)
  rho = np.where(u > (sv - b) / range(1, len(u) + 1))[0][-1]
  theta = max(0, (sv[rho] - b) / (rho + 1)) # maximum amount by which to shrink entries
  # print(theta)
  # print(abs(v) - theta)
  # print((abs(v) - theta).shape)
  # print(np.zeros(v.shape).shape)
  return np.sign(v) * np.maximum(abs(v) - theta, np.zeros(v.shape))

def multiconv(A, B):

  (Na, Wa) = np.shape(A);
  (Nb, Wb) = np.shape(B);


  if not Wa == Wb:
    raise ValueError(['A and B must have the same width. A has width ' + str(Wa) + \
                      ' and B has width ' + str(Wb) + '.']);

  C = np.zeros(Na + Nb - 1)
  for col_idx in range(Wa):
    C += np.convolve(A[:, col_idx], B[:, col_idx])
  return C
