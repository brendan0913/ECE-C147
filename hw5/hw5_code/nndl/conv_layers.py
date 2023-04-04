import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
    
  H_prime = 1 + (H + 2 * pad - HH) / stride
  W_prime = 1 + (W + 2 * pad - WW) / stride
  H_prime, W_prime = np.int(H_prime), np.int(W_prime)
    
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad, pad)))
  out = np.zeros((N, F, H_prime, W_prime))
  
  for img in range(N):
      for r in range(H_prime):
        for c in range(W_prime):
          stride_r, stride_c = r*stride, c*stride
          x_seg = xpad[img, :, stride_r : stride_r+HH, stride_c : stride_c+WW]
          out[img, :, r, c] = np.sum(x_seg*w, axis=(1,2,3)) + b
  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  H, W = x.shape[2:]
    
  db = np.sum(dout, axis=(0, 2, 3)) # db is of size F
  
  dw = np.zeros(w.shape)
  dx_pad = np.zeros(xpad.shape)
  for img in range(N):
    for r in range(out_height):
      for c in range(out_width):
        for filt in range(num_filts):
          stride_r, stride_c = r*stride, c*stride
          x_seg = xpad[img, :, stride_r : stride_r+f_height, stride_c : stride_c+f_width]
          dw[filt] += dout[img, filt, r, c] * x_seg
          dx_pad[img, :, stride_r : stride_r+f_height, stride_c : stride_c+f_width] += dout[img, filt, r ,c] * w[filt]
        
  dx = dx_pad[..., pad : H+pad, pad : W+pad]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    
  N, C, H, W = x.shape
  out_height = (H - pool_height) / stride + 1
  out_width = (W - pool_width) / stride + 1
  out_height, out_width = np.int(out_height), np.int(out_width)
  out = np.zeros((N, C, out_height, out_width))

  for img in range(N):
    for channel in range(C):
      for r in range(out_height):
        for c in range(out_width):
          stride_r, stride_c = r*stride, c*stride
          out[img, channel, r, c] = np.max(x[img, channel, stride_r : stride_r+pool_height, stride_c : stride_c+pool_width])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  dout_height, dout_width = dout.shape[2:]
  dx = np.zeros(x.shape)

  for img in range(N):
    for channel in range(C):
      for r in range(dout_height):
        for c in range(dout_width):
          stride_r, stride_c = r*stride, c*stride
          window = x[img, channel, stride_r : stride_r + pool_height, stride_c : stride_c + pool_width]
          d = dout[img, channel, r, c] * (window == np.max(window))
          dx[img, channel, stride_r : stride_r + pool_height, stride_c : stride_c + pool_width] = d
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = x.shape
  x_transpose = x.transpose(0, 2, 3, 1) # N, H, W, C
  x_reshape = x_transpose.reshape(N*H*W, C)
  out, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param) 
  out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2) # N, C, H, W
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout_transpose = dout.transpose(0, 2, 3, 1) # N, H, W, C
  dout_reshape = dout_transpose.reshape(N*H*W, C)
  dx, dgamma, dbeta = batchnorm_backward(dout_reshape, cache)
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2) # N, C, H, W

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta