'''
Author: Gyan Mittal
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import imageio
from util import *

def naive_softmax(x):
    return np.exp(x)/np.exp(x).sum(axis=1).reshape(-1,1)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def predict(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    y_pred = naive_softmax(Z2)
    y_pred = np.array(np.argmax(y_pred, axis=1))
    return y_pred

def accuracy(y, y_pred):
    acc = int(sum(y == y_pred) / len(y) * 100)
    return acc

def back_propagation(train_x, train_y_one_hot_vector, yhat, Z1, A1, W1, b1, W2, b2, learning_rate=0.0005):

    #dl_dyhat = np.divide(train_y_one_hot_vector, pred_train_y)
    dl_dz2 = yhat - train_y_one_hot_vector
    dl_dA1 = dl_dz2.dot(W2.T)
    dl_dw2 = A1.T.dot(dl_dz2)
    #dl_b2 = np.sum(dl_dz2, axis=0, keepdims=True)
    dl_b2 = np.sum(dl_dz2, axis=0)

    dl_dz1 = dl_dA1 * d_relu(Z1)
    dl_dw1 = train_x.T.dot(dl_dz1)
    #dl_db1 = np.sum(dl_dz1, axis=0, keepdims=True)
    dl_db1 = np.sum(dl_dz1, axis=0)

    # update the weights and bias
    W1 -= learning_rate * dl_dw1
    W2 -= learning_rate * dl_dw2
    b1 -= learning_rate * dl_db1
    b2 -= learning_rate * dl_b2

    return W1, W2, b1, b2

def forward_propagation(train_x, W1, b1, W2, b2):
    Z1 = train_x.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    yhat = naive_softmax(Z2)
    return Z1, A1, yhat

def cross_entropy_loss(y, y_pred):
    n_samples = len(y)
    loss = (-1/n_samples) * (np.sum(np.multiply(np.log(y_pred), y)))
    return loss

def initiate_weights(input_size, hidden_layer_size, output_size):
    np.random.seed(100)
    W1 = np.random.randn(input_size, hidden_layer_size)
    b1 = np.random.randn(hidden_layer_size, )
    W2 = np.random.randn(hidden_layer_size, output_size)
    b2 = np.random.randn(output_size, )
    return W1, b1, W2, b2

filenames = []
def plot_separation_line_and_loss(train_x, train_y, W1, b1, W2, b2, loss, epoch, max_loss):

    y_pred = predict(train_x, W1, b1, W2, b2)
    train_accuracy = accuracy(train_y, y_pred)

    x1 = np.arange(min(train_x[:, 0]) - 0.5, max(train_x[:, 0]) + 0.5, .1)
    x2 = np.arange(min(train_x[:, 1]) - 0.5, max(train_x[:, 1]) + 0.5, .1)

    X = np.array([[a, b] for a in x1 for b in x2])
    y_pred = predict(X, W1, b1, W2, b2)

    # Plotting
    #fig = plt.figure(figsize=(9, 9))
    fig, (ax1, ax2) = plt.subplots(1,2, sharex=False, figsize=(18, 9))

    train_y = train_y.flatten()
    ax1.plot(train_x[:,0][train_y==0], train_x[:,1][train_y==0], "cD")
    ax1.plot(train_x[:,0][train_y==1], train_x[:,1][train_y==1], "mD")
    ax1.plot(train_x[:,0][train_y==2], train_x[:,1][train_y== 2], "bD")

    ax1.set_title('Multi Class Classification training data')
    plt.setp(ax1, xlabel='pred_x1', ylabel='pred_x2')
    ax1.set_xlim([min(train_x[:, 0]) - 1, max(train_x[:, 0]) + 1])
    ax1.set_ylim([min(train_x[:, 1]) - 1, max(train_x[:, 1]) + 1])

    ax2.set_title("Loss graph")
    plt.setp(ax2, xlabel='#Epochs (Log scale)', ylabel='Loss')
    ax2.set_xlim([1 , epoch * 1.1])
    ax2.set_xscale('log')
    ax2.set_ylim([-0.1, max_loss * 1.1])

    if(len(loss) > 0):
        ax1.plot(X[:, 0][y_pred == 0], X[:, 1][y_pred == 0], "c.")
        ax1.plot(X[:, 0][y_pred == 1], X[:, 1][y_pred == 1], "m.")
        ax1.plot(X[:, 0][y_pred == 2], X[:, 1][y_pred == 2], "b.")
        ax1.set_title('Multi Class Classification with ' + str(train_accuracy) + '% accuracy after ' + str(f'{len(loss)-1:,}') + ' epochs')

        ax2.plot(loss)
        ax2.set_title("Loss after " + str(f'{len(loss)-1:,}') + " epochs is " + str("{:.6f}".format(loss[-1])))


    directory = "images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'images/{len(loss)}.png'
    for i in range(10):
        filenames.append(filename)
    # save frame
    plt.savefig(filename)
    plt.close()

'''
#multi class classification [linear]
train_X = np.array([[0.5,1], [0.5,2], [1,2],[1,3], [2,3], [3,5], [3,6], [1,3], [4,3],[5,4], [6,5], [8,6], [5,3], [6,3], [7,4], [8,5], [3,1], [3,2], [4,1], [4,2], [6,1], [6,2], [5,1], [5,2]])
train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2])
'''

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

#'''
# Training data for the binary classification with rectangular boundary
np.random.seed(100)
train_x = np.random.rand(100, 2) * 5
train_y = []
for i, val in enumerate(train_x):
    if (math.fabs(val[0] - 2.5) < 1.95 and math.fabs(val[1] - 2.5) < 1.95):
        train_y.append(0)
    else:
        train_y.append(1)
train_y = np.array(train_y)
#'''

no_of_classes = len(np.unique(train_y))
train_y_one_hot_vector = np.array([np.zeros(no_of_classes)] * len(train_y))
for idx, val in enumerate(train_y):
    train_y_one_hot_vector[idx][val] = 1

# Neural network with one hidden size
input_size = len(train_x[0]) # Number of input features
hidden_layer_size = 20 # Design choice
output_size = no_of_classes # Number of classes

W1, b1, W2, b2 = initiate_weights(input_size, hidden_layer_size, output_size)
epoch = 20000
loss_log =[]

saved_epoch_no = 0
for epoch_no in range(epoch):
    loss = 0
    Z1, A1, yhat = forward_propagation(train_x, W1, b1, W2, b2)
    loss = cross_entropy_loss(train_y_one_hot_vector, yhat)
    if (epoch_no==0):
        plot_separation_line_and_loss(train_x, train_y, W1, b1, W2, b2, loss_log, epoch, loss)
        loss_log.append(loss)
    loss_log.append(loss)

    W1, W2, b1, b2 = back_propagation(train_x, train_y_one_hot_vector, yhat, Z1, A1, W1, b1, W2, b2)

    if ((epoch_no == 1) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        y_pred = predict(train_x, W1, b1, W2, b2)
        print("epoch_no: ", (epoch_no + 1), "\tloss_log:", loss, "\taccuracy:", accuracy(train_y, y_pred))
        if (epoch_no >= 1): plot_separation_line_and_loss(train_x, train_y, W1, b1, W2, b2, loss_log, epoch, max(loss_log))
        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))

create_gif(filenames, 'images/neural_network_multi_class_classification.gif')