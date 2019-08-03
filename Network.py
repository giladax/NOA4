import Softmax as sm
import SGD as sgd
import Tanh as th

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

        self._theta_layer_size = self._sample_size + (self._sample_size ** 2)
        last_layer_size = self._num_labels * (self._sample_size + 1)
        self.Theta = np.random.randn(self._num_layers * self._theta_layer_size + last_layer_size)

    def get_layer_weights(self, k, sample_size):
        layer_weights = self.Theta[k * self._theta_layer_size: (k + 1) * self._theta_layer_size]

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

            prev_x = X_layer_out[:, :, k]

            X_layer_out[:, :, k + 1] = th.tanh(X_layer_out[:, :, k], W, b)

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

        grad_x_loss = sm.grad_x(X_layer_out[:, :, self._num_layers], W_loss, b_loss, self.C)

        for k in range(self._num_layers - 1, -1, -1):
            W, b = self.get_layer_weights(k, self._sample_size)
            grad_x_loss = W * grad_x_loss

            curr_x = X_layer_out[:, :, k]

            grad_w_layer = th.grad_tanh(curr_x, W, b)

            vl = grad_x_loss * grad_w_layer
            der_ce_b = np.mean(vl, axis=1)
            der_ce_w = vl * np.transpose(curr_x)
            grad_theta[(k - 1) * self._theta_layer_size: k * self._theta_layer_size] = np.hstack(
                (der_ce_b, der_ce_w.flatten()))

        return grad_theta

    def fit_net(self):

