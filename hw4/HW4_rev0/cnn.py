import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Your initializations should work for any valid input dims,      #
        # number of filters, hidden dims, and num_classes. Assume that we use      #
        # max pooling with pool height and width 2 with stride 2.                  #
        #                                                                          #
        # For Linear layers, weights and biases should be initialized from a       #
        # uniform distribution from -sqrt(k) to sqrt(k),                           #
        # where k = 1 / (#input features)                                          #
        # For Conv. layers, weights should be initialized from a uniform           #
        # distribution from -sqrt(k) to sqrt(k),                                   #
        # where k = 1 / ((#input channels) * filter_size^2)                        #
        # Note: we use the same initialization as pytorch.                         #
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html           #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html           #
        #                                                                          #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights for the convolutional layer using the keys 'W1' and 'W2'   #
        # (here we do not consider the bias term in the convolutional layer);      #
        # use keys 'W3' and 'b3' for the weights and biases of the                 #
        # hidden fully-connected layer, and keys 'W4' and 'b4' for the weights     #
        # and biases of the output affine layer.                                   #
        #                                                                          #
        # Make sure you have initialized W1, W2, W3, W4, b3, and b4 in the         #
        # params dicitionary.                                                      #
        #                                                                          #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3. Calculate the size of W3 dynamically           #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        # Calculate the size after the first conv and pool layers
        H1 = (self.H - filter_size + 1) // 2
        W1 = (self.W - filter_size + 1) // 2

        # Calculate the size after the second conv and pool layers
        H2 = (H1 - filter_size + 1) // 2
        W2 = (W1 - filter_size + 1) // 2

        # Convolutional layers
        self.params['W1'] = np.random.uniform(-np.sqrt(1/(self.C*filter_size**2)), np.sqrt(1/(self.C*filter_size**2)), (num_filters_1, self.C, filter_size, filter_size))
        self.params['W2'] = np.random.uniform(-np.sqrt(1/(num_filters_1*filter_size**2)), np.sqrt(1/(num_filters_1*filter_size**2)), (num_filters_2, num_filters_1, filter_size, filter_size))

        # Fully connected layers
        self.params['W3'] = np.random.uniform(-np.sqrt(1/(num_filters_2*H2*W2)), np.sqrt(1/(num_filters_2*H2*W2)), (num_filters_2*H2*W2, hidden_dim))
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, num_classes))
        self.params['b4'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3.                                                #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        # Conv1 -> ReLU -> Pool1
        out_conv1, cache_conv1 = conv_forward(X, W1)
        out_relu1, cache_relu1 = relu_forward(out_conv1)
        out_pool1, cache_pool1 = max_pool_forward(out_relu1, pool_param)

        # Conv2 -> ReLU -> Pool2
        out_conv2, cache_conv2 = conv_forward(out_pool1, W2)
        out_relu2, cache_relu2 = relu_forward(out_conv2)
        out_pool2, cache_pool2 = max_pool_forward(out_relu2, pool_param)

        # Flatten the output from the pooling layer to make it a vector
        out_pool2_flat = out_pool2.reshape(X.shape[0], -1)

        # FC1 -> ReLU
        out_fc1, cache_fc1 = fc_forward(out_pool2_flat, W3, b3)
        out_relu3, cache_relu3 = relu_forward(out_fc1)

        # FC2 -> Softmax
        scores, cache_fc2 = fc_forward(out_relu3, W4, b4)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k].                                                      #
        # Hint: The backwards from W3 needs to be un-flattened before it can be    #
        # passed into the max pool backwards                                       #
        ############################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        # Start with the gradient from the loss
        loss, dscores = softmax_loss(scores, y)

        # Backprop through FC2
        dx, dw4, db4 = fc_backward(dscores, cache_fc2)
        grads['W4'] = dw4
        grads['b4'] = db4

        # Backprop through ReLU3
        dx = relu_backward(dx, cache_relu3)

        # Backprop through FC1
        dx, dw3, db3 = fc_backward(dx, cache_fc1)
        grads['W3'] = dw3
        grads['b3'] = db3

        # Un-flatten dx to match the shape after max pooling 2
        dx = dx.reshape(out_pool2.shape)

        # Backprop through Pool2
        dx = max_pool_backward(dx, cache_pool2)

        # Backprop through ReLU2
        dx = relu_backward(dx, cache_relu2)

        # Backprop through Conv2
        dx, dw2 = conv_backward(dx, cache_conv2)
        grads['W2'] = dw2

        # Backprop through Pool1
        dx = max_pool_backward(dx, cache_pool1)

        # Backprop through ReLU1
        dx = relu_backward(dx, cache_relu1)

        # Backprop through Conv1
        dx, dw1 = conv_backward(dx, cache_conv1)
        grads['W1'] = dw1

        return loss, grads


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
