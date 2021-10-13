'''
Author: Gyan Mittal
'''
import numpy as np
from util import *

def predict(X, W, b):
    Z = X.dot(W) + b
    y_pred = naive_softmax(Z)
    y_pred = np.array(np.argmax(y_pred, axis=1))
    return y_pred

def accuracy(y, y_pred):
    acc = int(sum(y == y_pred) / len(y) * 100)
    return acc

def back_propagation(train_x, train_y_one_hot_vector, yhat, W, b, learning_rate):
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

def cross_entropy_loss(y, y_pred):
    n_samples = len(y)
    loss = (-1/n_samples) * (np.sum(np.multiply(np.log(y_pred), y)))
    return loss

def initiate_weights(input_size, output_size):
    np.random.seed(1)
    W = np.random.randn(input_size, output_size)
    b = np.random.randn(output_size, )
    return W, b


#'''
#multi class classification [linear]
train_X = np.array([[0.5, 1], [0.5, 2], [1, 2], [1, 3], [2, 3], [3, 5], [3, 6], [1, 3], [4, 3], [5, 4], [6, 5], [8, 6], [5, 3], [6, 3], [7, 4], [8, 5], [3, 1], [3, 2], [4, 1], [4, 2], [6, 1], [6, 2], [5, 1], [5, 2]])
train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2])
#'''

'''
#binary classification [linear]
train_X = np.array([[0.5,1], [0.5,2], [1,2],[1,3], [2,3], [3,5], [3,6], [1,3], [4,3],[5,4], [6,5], [8,6], [5,3], [6,3], [7,4], [8,5]])
train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
'''

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

# Neural network with one hidden size
input_size = len(train_X[0]) # Number of input features
output_size = no_of_classes # Number of classes

W, b = initiate_weights(input_size, output_size)

pred_x1 = np.arange(min(train_X[:, 0]) - 0.5, max(train_X[:, 0]) + 0.5, .1)
pred_x2 = np.arange(min(train_X[:, 1]) - 0.5, max(train_X[:, 1]) + 0.5, .1)
pred_X = np.array([[x1, x2] for x1 in pred_x1 for x2 in pred_x2])

epoch = 200000
learning_rate = 0.001
loss_log =[]

saved_epoch_no = 0
for epoch_no in range(epoch):
    loss = 0
    yhat = forward_propagation(train_X, W, b)
    loss = cross_entropy_loss(train_y_one_hot_vector, yhat)
    if (epoch_no==0):
        pred_train_y = predict(train_X, W, b).flatten()
        pred_y = predict(pred_X, W, b).flatten()
        image_files = plot_classification_separation_line_and_loss(train_X, train_y, pred_train_y, pred_X, pred_y, loss_log, epoch, max_loss=loss)
        loss_log.append(loss)
    loss_log.append(loss)

    W, b1 = back_propagation(train_X, train_y_one_hot_vector, yhat, W, b, learning_rate)

    if ((epoch_no == 1) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        y_pred = predict(train_X, W, b)
        print("epoch_no: ", (epoch_no + 1), "\tloss_log:", loss, "\taccuracy:", accuracy(train_y, y_pred))
        if (epoch_no >= 1):
            pred_train_y = predict(train_X, W, b).flatten()
            pred_y = predict(pred_X, W, b).flatten()
            image_files = plot_classification_separation_line_and_loss(train_X, train_y, pred_train_y, pred_X, pred_y, loss_log, epoch, max(loss_log), image_files)
        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))

create_gif(image_files, 'images/logistic_regression_multiclass_classification.gif')