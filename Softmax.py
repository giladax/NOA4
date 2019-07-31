import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


def grad_x(X, W, b, C):
    _, m = X.shape
    softmax_result = softmax(X, W, b)
    return W.dot(softmax_result - C) / m


def grad_theta(X, W, b, C):
    _, m = X.shape
    softmax_result = softmax(X, W, b)
    g_W = X.dot(softmax_result.T - C.T)
    g_b = np.sum(softmax_result.T - C.T, 0)
    return np.hstack((g_b, g_W.flatten())) / m


def softmax(x, W, b):
    xT_W = W.T.dot(x) + b
    max_elment = np.argmax(xT_W, axis=0)
    numerator = np.exp(xT_W - max_elment)
    denominator = numerator.sum(axis=0)
    result = (numerator / denominator)
    return result


def loss(X, W, b, C, eps=1e-9):
    _, m = X.shape
    softmax_result = softmax(X, W, b)
    cal_loss = np.multiply(np.log(softmax_result + eps), C)
    return -np.sum(cal_loss) / m


def gradient_test(f, grad_f, X, W, b, C, max_iter, test_indicator, to_print=False):
    d, samples = X.shape
    n_labels, samples = C.shape

    d_x = np.random.rand(d, samples) * test_indicator[0]
    d_w = np.random.rand(d, n_labels) * test_indicator[1]
    d_b = np.random.rand(n_labels, 1) * test_indicator[2]

    f_test = []  # The linear test
    g_test = []  # The quadratic test

    # define the limits and vectors according the test
    lower_limit = 0
    upper_limit = None

    if (test_indicator[1]):
        lower_limit = n_labels
        d_b, d_x = 0, 0

    elif (test_indicator[2]):
        upper_limit = n_labels
        d_w, d_x = 0, 0
    else:
        d_b, d_w = 0, 0

    eps = 1
    for i in range(max_iter):
        eps = eps * 0.5;

        f_value = f(X, W, b, C)
        curr_dx, curr_dw, curr_db = eps * d_x, eps * d_w, eps * d_b

        f_value_diff = f(X + curr_dx, W + curr_dw, b + curr_db, C)
        f_diff = abs(f_value_diff - f_value)
        f_test.append(f_diff)

        curr_test_vec = curr_dx * test_indicator[0] + curr_dw * test_indicator[1] + curr_db * test_indicator[2]

        if (test_indicator[0]):
            g_diff = abs(f_value_diff - f_value - np.dot(curr_test_vec.flatten(), grad_f(X, W, b, C).flatten()))
        else:
            g_diff = abs(
                f_value_diff - f_value - np.dot(curr_test_vec.flatten(), grad_f(X, W, b, C)[lower_limit:upper_limit]))
        g_test.append(g_diff / eps)

        if (to_print and i > 1 and i < max_iter):
            print("Linear measure: {}".format(f_test[i]))
            print("Linear diff: {}\n".format(f_test[i] / f_test[i - 1]))

            print("Quad measure: {}".format(g_test[i]))
            print("Quad diff: {}\n".format(g_test[i] / g_test[i - 1]))

    return f_test, g_test


def run_q():
    data = sio.loadmat(r'SwissRollData.mat')
    X, C = data['Yt'], data['Ct']
    iter_number = 20

    d, samples = X.shape
    labels, samples = C.shape

    W = np.random.rand(d, labels)
    b = np.random.rand(labels, 1)

    # Thete test
    X_test_indicator = [1, 0, 0]
    theta_test_indicator = [0, 1, 0]
    b_test_indicator = [0, 0, 1]

    f_tes_x, g_test_x = gradient_test(loss, grad_x, X, W, b, C, iter_number, X_test_indicator, to_print=True)
    # f_tes_w, g_test_w = gradient_test(loss, grad_theta, X,W,b, C, iter_number, theta_test_indicator, to_print=True)
    # f_tes_b, g_test_b = gradient_test(loss, grad_theta, X,W,b, C, iter_number, b_test_indicator, to_print=True)

    plt.subplot(2, 2, 1)
    plt.title('Linear gradient test')
    plt.semilogy(f_tes_x, 'r', label='Linear')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title('Quadratic gradient test value divided by epsilon ')
    plt.semilogy(g_test_x, 'b', label='Quadratic')
    plt.xlabel('Value')
    plt.legend()
    plt.show()
