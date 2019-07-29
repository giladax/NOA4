import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


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


def grad_theta(X, W, b, C):
    _, m = X.shape
    softmax_result = softmax(X, W, b)
    g_W = X.dot(softmax_result.T - C.T)
    g_b = np.sum(softmax_result.T - C.T, 0)
    return np.hstack((g_b, g_W.flatten())) / m


def calculate_accuracy(X, W, b, C):
    number_labels, number_samples = X.shape
    softmax_result = softmax(X, W, b)
    predicted_labels = np.argmax(softmax_result, 0)
    correct_labels = np.argmax(C, axis=0)
    match = sum(np.equal(predicted_labels, correct_labels))
    return 100 * (match / number_samples)


def train_SGD(X, C, theta, b, X_validate, C_validate, max_epoch=5):
    sample_size, number_samples = X.shape
    labels, number_samples = C.shape
    batch_size = 25
    lr = 0.25
    lr_update = 50
    lr_update_rate = 0.9
    validate_every = 1
    training_loss, validation_loss, training_accuracy, validation_accuracy = [], [], [], []

    for epoch in range(max_epoch):

        # Shuffle the data
        random_indexes = np.random.permutation(number_samples)
        X_shuffeled = X[:, random_indexes]
        C_shuffeled = C[:, random_indexes]

        if (epoch % lr_update == 0):
            lr *= lr_update_rate

        for i in range(int(number_samples / batch_size)):
            first_index = i * batch_size
            last_index = (i + 1) * batch_size
            if (last_index >= number_samples):
                last_index = number_samples - 1

            batch_X = X_shuffeled[:, first_index:last_index]
            batch_C = C_shuffeled[:, first_index:last_index]
            g_k = grad_theta(batch_X, theta, b, batch_C)

            b = b - lr * np.reshape(g_k[:labels], (b.shape))
            theta = theta - lr * np.reshape(g_k[labels:], (theta.shape))

        # Analyse results
        if (epoch % validate_every == 0):
            loss_val = loss(X_validate, theta, b, C_validate)
            validation_loss.append(loss_val)

            accuracy_val = calculate_accuracy(X_validate, theta, b, C_validate)
            validation_accuracy.append(accuracy_val)

            loss_trn = loss(X, theta, b, C)
            training_loss.append(loss_trn)
            accuracy_trn = calculate_accuracy(X, theta, b, C)
            training_accuracy.append(accuracy_trn)

            print("The batch loss: {}".format(loss_val))
            print("The batch acuracy: {}\n".format(accuracy_val))

    return theta, training_accuracy, training_loss, validation_accuracy, validation_loss


def run_q3():
    mat = sio.loadmat("GMMData.mat")

    ct = mat.get("Ct")
    xt = mat.get("Yt")
    cv = mat.get("Cv")
    xv = mat.get("Yv")

    num_labels, _ = ct.shape
    sample_size, num_train_samples = xt.shape

    W = np.random.randn(sample_size, num_labels)
    b = np.random.rand(num_labels, 1)
    theta, training_accuracy, training_loss, validation_accuracy, validation_loss = train_SGD(xt, ct, W, b, xv, cv, 150)
    return training_accuracy, training_loss, validation_accuracy, validation_loss


training_accuracy, training_loss, validation_accuracy, validation_loss = run_q3()
plt.subplot(2, 2, 1)
plt.title('SGD Softmax Train Loss -GMMData')
plt.semilogy(training_loss, 'r', label='Train')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.title('SGD Softmax Validation Loss-GMMData')
plt.semilogy(validation_loss, 'b', label='Validation')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(2, 2, 3)
plt.title('SGD Softmax Train Accuracy-GMMData')
plt.semilogy(training_accuracy, 'r', label='Train')
plt.xlabel('Epoch')
plt.ylabel('Success ratio')
plt.legend()

plt.subplot(2, 2, 4)
plt.title('SGD Softmax Validation Accuracy-GMMData')
plt.semilogy(validation_accuracy, 'b', label='Validation')
plt.xlabel('Epoch')
plt.legend()

plt.show()
