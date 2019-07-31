import Softmax as sm
import SGD as sgd

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


class Network:
    def __init__(self, nun_layers, X, C, eps=0.1, max_epoc=100):
        self._num_layers = nun_layers
        self.X = X
        self.C = C
        self.eps = eps
        self.max_epoc = max_epoc

        self._sample_size, self._num_samples = self.X.shape
        self._num_labels, self._number_samples = self.C.shape

        # Initialize a random net
        d, _ = X.shape
        num_labels, _ = C.shape

        self._theta_layer_size = d + (d ** 2)
        last_layer_size = num_labels * (d + 1)
        self.Theta = np.random.randn(self._num_layers * self._theta_layer_size + last_layer_size)

    def get_layer_weights(self, k, sample_size):
        theta_layer_size = sample_size + (sample_size ** 2)
        layer_weights = self.Theta[k * theta_layer_size: (k + 1) * theta_layer_size]

        b = layer_weights[: sample_size]
        W = layer_weights[sample_size: sample_size + sample_size ** 2]

        return W.reshape(sample_size, sample_size), b

    def grad_x_loss(self, X, W, b):
        o = sm.softmax(X, W, b)

        return o - self.C

    def grad_last_layer(self, x_last_layer, W, b):
        o = sm.softmax(x_last_layer, W, b)

        der_c_e = np.transpose(self.C) - np.transpose(o)
        der_b = np.mean(der_c_e, axis=1)
        der_w = (x_last_layer * der_c_e) / self._num_samples

        return np.concatenate(der_b, der_w.flatten())

    def forward_pass(self):
        # sample_size, num_samples = self.X.shape

        X_layer_out = np.zeros((self._sample_size, self._num_samples, self._num_layers + 1))
        X_layer_out[:, :, 0] = self.X  # first Xis is the input layer

        for k in range(self._num_layers):
            [W, b] = self.get_layer_weights(k, self._sample_size)
            X_layer_out[:, :, k + 1] = sm.softmax(X_layer_out[:, :, k], W, b)

        loss_weights_idx = self._num_layers * ((self._sample_size ** 2) + self._sample_size)
        b = self.Theta[loss_weights_idx: loss_weights_idx + self._num_labels]
        W = self.Theta[
            loss_weights_idx + self._num_labels: loss_weights_idx + self._num_labels + self._sample_size * self._num_labels]
        W = W.reshape(self._sample_size, self._num_labels)

        # Compute loss and probabilities
        all_prob = sm.softmax(X_layer_out[:, :, self._num_layers], W, b)
        relevant_prob = self.C * np.log(all_prob + self.eps)
        loss = - sum(relevant_prob[:])

        return all_prob, loss, X_layer_out

    def backward_pass(self, X_layer_out):
        # sample_size, num_samples = self.X.shape
        # num_labels, _ = self.C.shape

        # Extract loss weights from Theta
        loss_weights_idx = self._num_layers * ((self._sample_size ** 2) + self._sample_size)
        loss_weights_end_loc = loss_weights_idx + self._num_layers + self._sample_size * self._num_labels

        b_loss = self.Theta[loss_weights_idx: loss_weights_idx + self._num_labels]
        W_loss = self.Theta[
                 loss_weights_idx + self._num_labels: loss_weights_idx + self._num_labels + self._sample_size * self._num_labels]
        W_loss = W_loss.reshape(self._sample_size, self._num_labels)

        grad_theta = np.zeros(self.Theta.shape)

        grad_w_loss = self.grad_last_layer(X_layer_out[:, :, self._num_layers], W_loss, b_loss)
        grad_theta[loss_weights_idx: loss_weights_end_loc] = grad_w_loss
        grad_x_loss = self.grad_x_loss(X_layer_out[:,:, self._num_layers], W_loss, b_loss)

        for k in range(self._num_layers - 1, -1, -1):
            W, b = self.get_layer_weights(k, self._sample_size)
            grad_x_loss = W * grad_x_loss

            g_w_layer = ResNN_jac_theta_t_mul(X_layer_out[:,:, k], W, b, grad_x_loss)
            grad_theta((k - 1) * theta_layer_size + 1: k * theta_layer_size) = g_w_layer;

            G_x = ResNN_jac_x_t_mul(Xis(:,:, k), W1, W2, b, G_x);

        return grad_theta


