import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    C, H, W = input_dim
    stride = 1
    pad = (filter_size - 1) / 2
    
    output_conv_height = (H + 2*pad - filter_size) / stride + 1
    output_conv_width = (W + 2*pad - filter_size) / stride + 1
    
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    
    # 2x2 max pool, stride 2
    output_pool_height = int((output_conv_height - 2) / 2 + 1)
    output_pool_width = int((output_conv_height - 2) / 2 + 1)
    
    pool_size = output_pool_height * output_pool_width * num_filters
    self.params['W2'] = np.random.normal(0, weight_scale, (pool_size, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    
    self.bn_params = {}
    if self.use_batchnorm:
        self.bn_params['bn_param1'] = {'mode': 'train'}
        self.params['beta1'] = np.zeros(num_filters)
        self.params['gamma1'] = np.ones(num_filters)

        self.bn_params['bn_param2'] = {'mode': 'train'}
        self.params['beta2'] = np.zeros(hidden_dim)
        self.params['gamma2'] = np.ones(hidden_dim)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    if self.use_batchnorm:
        bn_param1 = self.bn_params['bn_param1']
        bn_param2 = self.bn_params['bn_param2']

        beta1 = self.params['beta1']
        beta2 = self.params['beta2']

        gamma1 = self.params['gamma1']
        gamma2 = self.params['gamma2']
        
        c_out, c_cache = conv_batchnorm_relu_pool_forward(X, W1, b1, conv_param, pool_param, gamma1, beta1, bn_param1)
        N, F, H, W = c_out.shape
        
        c_out = c_out.reshape((N, F*H*W))
        ar_out, ar_cache = affine_batchnorm_relu_forward(c_out, W2, b2, gamma2, beta2, bn_param2)
    else:  
        c_out, c_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        ar_out, ar_cache = affine_relu_forward(c_out, W2, b2)
        
    scores, scores_cache = affine_forward(ar_out, W3, b3)
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, grad_loss = softmax_loss(scores, y)
    
    loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

    dx, dw3, grads['b3'] = affine_backward(grad_loss, scores_cache)
    
    if self.use_batchnorm:
        dx, dw2, grads['b2'], grads['gamma2'], grads['beta2'] = affine_batchnorm_relu_backward(dx, ar_cache)
        dx = dx.reshape((N, F, H, W))
        dx, dw1, grads['b1'], grads['gamma1'], grads['beta1'] = conv_batchnorm_relu_pool_backward(dx, c_cache)
    else:
        dx, dw2, grads['b2'] = affine_relu_backward(dx, ar_cache)
        dx, dw1, grads['b1'] = conv_relu_pool_backward(dx, c_cache)

    grads['W3'] = dw3 + self.reg * W3
    grads['W2'] = dw2 + self.reg * W2
    grads['W1'] = dw1 + self.reg * W1

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
