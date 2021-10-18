'''
Author: Gyan Mittal
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import imageio
from util import *

def predict(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    y_pred = naive_softmax(Z2)
    y_pred = np.array(np.argmax(y_pred, axis=1))
    return y_pred

def back_propagation(train_X, train_y_one_hot_vector, yhat, Z1, A1, W1, b1, W2, b2, learning_rate=0.0005):

    #dl_dyhat = np.divide(train_y_one_hot_vector, pred_train_y)
    dl_dz2 = yhat - train_y_one_hot_vector
    dl_dA1 = dl_dz2.dot(W2.T)
    dl_dw2 = A1.T.dot(dl_dz2)
    #dl_b2 = np.sum(dl_dz2, axis=0, keepdims=True)
    dl_b2 = np.sum(dl_dz2, axis=0)

    dl_dz1 = dl_dA1 * d_relu(Z1)
    dl_dw1 = train_X.T.dot(dl_dz1)
    #dl_db1 = np.sum(dl_dz1, axis=0, keepdims=True)
    dl_db1 = np.sum(dl_dz1, axis=0)

    # update the weights and bias
    W1 -= learning_rate * dl_dw1
    W2 -= learning_rate * dl_dw2
    b1 -= learning_rate * dl_db1
    b2 -= learning_rate * dl_b2

    return W1, W2, b1, b2

def forward_propagation(train_X, W1, b1, W2, b2):
    Z1 = train_X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    yhat = naive_softmax(Z2)
    return Z1, A1, yhat

def initiate_weights(input_size, hidden_layer_size, output_size):
    np.random.seed(100)
    W1 = np.random.randn(input_size, hidden_layer_size)
    b1 = np.random.randn(hidden_layer_size, )
    W2 = np.random.randn(hidden_layer_size, output_size)
    b2 = np.random.randn(output_size, )
    return W1, b1, W2, b2

'''
#multi class classification [linear]
train_X = np.array([[0.5,1], [0.5,2], [1,2],[1,3], [2,3], [3,5], [3,6], [1,3], [4,3],[5,4], [6,5], [8,6], [5,3], [6,3], [7,4], [8,5], [3,1], [3,2], [4,1], [4,2], [6,1], [6,2], [5,1], [5,2]])
train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2])
'''

#'''
#binary classification [linear]
train_X = np.array([[0.5,1], [0.5,2], [1,2],[1,3], [2,3], [3,5], [3,6], [1,3], [4,3],[5,4], [6,5], [8,6], [5,3], [6,3], [7,4], [8,5]])
train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
#'''

'''
# Training data for the binary classification with circular boundary
np.random.seed(100)
train_X = np.random.rand(200, 2) * 5
train_y = []
for i, val in enumerate(train_X):
    distance = math.sqrt((val[0]-2.5)**2 + (val[1]-2.5)**2)

    if (distance < 1.95):
        train_y.append(0)
    else:
        train_y.append(1)
train_y = np.array(train_y)
'''

'''
# Training data for the binary classification with rectangular boundary
np.random.seed(100)
train_X = np.random.rand(100, 2) * 5
train_y = []
for i, val in enumerate(train_X):
    if (math.fabs(val[0] - 2.5) < 1.95 and math.fabs(val[1] - 2.5) < 1.95):
        train_y.append(0)
    else:
        train_y.append(1)
train_y = np.array(train_y)
'''

no_of_classes = len(np.unique(train_y))
train_y_one_hot_vector = np.array([np.zeros(no_of_classes)] * len(train_y))
for idx, val in enumerate(train_y):
    train_y_one_hot_vector[idx][val] = 1

# Neural network with one hidden layer
input_size = len(train_X[0]) # Number of input features
hidden_layer_size = 20 # Design choice
output_size = no_of_classes # Number of classes

test_data_x1 = np.arange(min(train_X[:, 0]) - 0.5, max(train_X[:, 0]) + 0.5, .1)
test_data_x2 = np.arange(min(train_X[:, 1]) - 0.5, max(train_X[:, 1]) + 0.5, .1)
test_data_X = np.array([[x1, x2] for x1 in test_data_x1 for x2 in test_data_x2])

W1, b1, W2, b2 = initiate_weights(input_size, hidden_layer_size, output_size)

epoch = 200000
loss_log =[]

saved_epoch_no = 0
for epoch_no in range(epoch):
    loss = 0
    Z1, A1, yhat = forward_propagation(train_X, W1, b1, W2, b2)
    loss = cross_entropy_loss(train_y_one_hot_vector, yhat)
    if (epoch_no==0):
        pred_train_y = predict(train_X, W1, b1, W2, b2)
        pred_test_data_y = predict(test_data_X, W1, b1, W2, b2)
        image_files = plot_classification_separation_line_and_loss(train_X, train_y, pred_train_y, test_data_X, pred_test_data_y,
                                                                   loss_log, epoch, max_loss=loss)
        loss_log.append(loss)
    loss_log.append(loss)

    W1, W2, b1, b2 = back_propagation(train_X, train_y_one_hot_vector, yhat, Z1, A1, W1, b1, W2, b2)

    if ((epoch_no == 1) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        pred_train_y = predict(train_X, W1, b1, W2, b2)
        print("epoch_no: ", (epoch_no + 1), "\tloss_log:", loss, "\taccuracy:", accuracy(train_y, pred_train_y))
        if (epoch_no >= 1):
            pred_test_data_y = predict(test_data_X, W1, b1, W2, b2)
            image_files = plot_classification_separation_line_and_loss(train_X, train_y, pred_train_y, test_data_X,
                                                                       pred_test_data_y, loss_log, epoch, max(loss_log),
                                                                       image_files)
        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))

create_gif(image_files, 'images/neural_network_multi_class_classification.gif')