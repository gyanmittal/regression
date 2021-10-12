'''
Author: Gyan Mittal
naive_logistic_regression_gradient_descent_binary_classification.py

'''
import numpy as np
from util import *

# Training data
#train_y = wx + x2
train_X = np.array([[0.5, 1], [0.5, 2], [1, 2], [1, 3], [2, 3], [3, 5], [3, 6], [1, 3], [4, 3], [5, 4], [6, 5], [8, 6], [5, 3], [6, 3], [7, 4], [8, 5]])
train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]).reshape(-1,1)

input_size = len(train_X[0]) # Number of input features
output_size = len(np.unique(train_y[0]))
# Initialize W and x2 with some random number
np.random.seed(100)
W = np.random.randn(input_size, output_size)
b = np.random.randn(output_size, )
pred_x1 = np.arange(min(train_X[:, 0]) - 0.5, max(train_X[:, 0]) + 0.5, .1)
pred_x2 = np.arange(min(train_X[:, 1]) - 0.5, max(train_X[:, 1]) + 0.5, .1)

pred_X = np.array([[x1, x2] for x1 in pred_x1 for x2 in pred_x2])

def predict(X, W, b):
    h = np.dot(X, W) + b
    pred_y = np.rint(sigmoid(h))
    return pred_y

epoch = 200000
learning_rate = 0.001
loss_log =[]
saved_epoch_no = 0
for epoch_no in range(epoch):
    loss = 0
    m = len(train_y)
    h = np.dot(train_X, W) + b
    yhat = sigmoid(h)

    loss = (-1 / m) * np.sum((train_y * np.log(yhat) + (1 - train_y) * np.log(1 - yhat)))

    dldw = (1 / m) * np.sum((yhat - train_y) * train_X, axis=0)
    dldw = np.array(dldw).reshape(-1,1)
    dldb = (1 / m) * np.sum((yhat - train_y))

    W -= learning_rate * dldw
    b -= learning_rate * dldb

    if (epoch_no==0):
        pred_train_y = predict(train_X, W, b).flatten()
        pred_y = predict(pred_X, W, b).flatten()
        image_files = plot_classification_separation_line_and_loss(train_X, train_y.flatten(), pred_train_y, pred_X, pred_y, loss_log, epoch, max_loss=loss)
        loss_log.append(loss)
    loss_log.append(loss)

    if ((epoch_no == 1) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        y_pred = predict(train_X, W, b)
        print("epoch_no: ", (epoch_no + 1), "\tloss_log:", loss, "\taccuracy:", accuracy(train_y, y_pred))
        if (epoch_no >= 1):
            pred_train_y = predict(train_X, W, b).flatten()
            pred_y = predict(pred_X, W, b).flatten()
            image_files = plot_classification_separation_line_and_loss(train_X, train_y.flatten(), pred_train_y, pred_X, pred_y, loss_log, epoch, max(loss_log), image_files)
        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))

create_gif(image_files, 'naive_logistic_regression_linear_binary_classification.gif')