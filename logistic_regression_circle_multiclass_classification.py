
'''
Author: Gyan Mittal
'''
import numpy as np
import math
from util import *

def predict(X, W, b):
    Z = X.dot(W) + b
    y_pred = naive_softmax(Z)
    y_pred = np.array(np.argmax(y_pred, axis=1))
    return y_pred

def back_propagation(train_x, train_y_one_hot_vector, yhat, W, b, learning_rate=0.001):

    dl_dz = yhat - train_y_one_hot_vector
    dl_dw = train_x.T.dot(dl_dz)
    dl_db = np.sum(dl_dz, axis=0)

    # update the weights and bias
    W -= learning_rate * dl_dw
    b -= learning_rate * dl_db

    return W, b

def forward_propagation(train_x, W, b):
    Z = train_x.dot(W) + b
    yhat = naive_softmax(Z)
    return yhat


def initiate_weights(input_size, output_size):
    np.random.seed(1)
    W = np.random.randn(input_size, output_size)
    b = np.random.randn(output_size, )
    return W, b


#'''
#multi class classification [linear]
#train_X = np.array([[0.5,1], [0.5,2], [1,2],[1,3], [2,3], [3,5], [3,6], [1,3], [4,3],[5,4], [6,5], [8,6], [5,3], [6,3], [7,4], [8,5], [3,1], [3,2], [4,1], [4,2], [6,1], [6,2], [5,1], [5,2]])
#actual_train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2])
#'''

'''
#binary classification [linear]
train_X = np.array([[0.5,1], [0.5,2], [1,2],[1,3], [2,3], [3,5], [3,6], [1,3], [4,3],[5,4], [6,5], [8,6], [5,3], [6,3], [7,4], [8,5]])
actual_train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
'''

#'''
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
train_X = np.array([[x1, x2, x1 * x1, x2 * x2] for x1, x2 in train_X])
#'''

'''
# Training data for the binary classification with rectangular boundary
np.random.seed(100)
train_X = np.random.rand(100, 2) * 5
actual_train_y = []
for i, val in enumerate(train_X):
    if (math.fabs(val[0] - 2.5) < 1.95 and math.fabs(val[1] - 2.5) < 1.95):
        actual_train_y.append(0)
    else:
        actual_train_y.append(1)
actual_train_y = np.array(actual_train_y)
'''

no_of_classes = len(np.unique(train_y))
train_y_one_hot_vector = np.array([np.zeros(no_of_classes)] * len(train_y))
for idx, val in enumerate(train_y):
    train_y_one_hot_vector[idx][val] = 1

# Neural network with one hidden size
input_size = len(train_X[0]) # Number of input features
output_size = no_of_classes # Number of classes

W, b = initiate_weights(input_size, output_size)

test_data_x1 = np.arange(min(train_X[:, 0]) - 0.5, max(train_X[:, 0]) + 0.5, .1)
test_data_x2 = np.arange(min(train_X[:, 1]) - 0.5, max(train_X[:, 1]) + 0.5, .1)
test_data_X = np.array([[x1, x2, x1 * x1, x2 * x2] for x1 in test_data_x1 for x2 in test_data_x2])

epoch = 200000
learning_rate = 0.0001
loss_log =[]

saved_epoch_no = 0

for epoch_no in range(epoch):
    loss = 0
    yhat = forward_propagation(train_X, W, b)
    loss = cross_entropy_loss(train_y_one_hot_vector, yhat)
    if (epoch_no==0):
        pred_train_y = predict(train_X, W, b)
        pred_test_data_y = predict(test_data_X, W, b)
        image_files = plot_classification_separation_line_and_loss(train_X, train_y, pred_train_y, test_data_X, pred_test_data_y, loss_log, epoch, max_loss=loss)
        loss_log.append(loss)
    loss_log.append(loss)

    W, b = back_propagation(train_X, train_y_one_hot_vector, yhat, W, b, learning_rate)

    if ((epoch_no == 1) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        pred_train_y = predict(train_X, W, b)
        print("epoch_no: ", (epoch_no + 1), "\tloss_log:", loss, "\taccuracy:", accuracy(train_y, pred_train_y))
        if (epoch_no >= 1):
            pred_train_y = predict(train_X, W, b)
            pred_test_data_y = predict(test_data_X, W, b)
            image_files = plot_classification_separation_line_and_loss(train_X, train_y, pred_train_y, test_data_X, pred_test_data_y, loss_log, epoch, max(loss_log), image_files)
        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))

create_gif(image_files, 'images/logistic_regression_circle_multiclass_classification.gif')

