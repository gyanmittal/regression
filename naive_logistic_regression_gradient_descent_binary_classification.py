'''
Author: Gyan Mittal
naive_logistic_regression_gradient_descent_binary_classification.py

'''
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from util import *

# Training data
#train_y = wx + b
train_x = np.array([[0.5,1], [0.5,2], [1,2],[1,3], [2,3], [3,5], [3,6], [1,3], [4,3],[5,4], [6,5], [8,6], [5,3], [6,3], [7,4], [8,5]])
train_y = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]).reshape(-1,1)

input_size = len(train_x[0]) # Number of input features
output_size = len(np.unique(train_y[0]))
# Initialize w and b with some random number
np.random.seed(100)
w = np.random.randn(input_size, output_size)
b = np.random.randn(output_size, )

def predict(X, w, b):
    h = np.dot(X, w) + b
    pred_y = np.rint(sigmoid(h))
    return pred_y

filenames = []
def plot_separation_line_and_loss(train_x, train_y, W, b, loss, epoch, max_loss):

    y_pred = predict(train_x, W, b)
    train_accuracy = accuracy(train_y, y_pred)

    x1 = np.arange(min(train_x[:, 0]) - 0.5, max(train_x[:, 0]) + 0.5, .1)
    x2 = np.arange(min(train_x[:, 1]) - 0.5, max(train_x[:, 1]) + 0.5, .1)

    X = np.array([[a, b] for a in x1 for b in x2])
    y_pred = predict(X, W, b)
    y_pred = y_pred.flatten()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1,2, sharex=False, figsize=(18, 9))

    train_y = train_y.flatten()
    ax1.plot(train_x[:,0][train_y==0], train_x[:,1][train_y==0], "cD")
    ax1.plot(train_x[:,0][train_y==1], train_x[:,1][train_y==1], "mD")
    ax1.plot(train_x[:,0][train_y==2], train_x[:,1][train_y== 2], "bD")

    ax1.set_title('Multi Class Classification training data')
    plt.setp(ax1, xlabel='x1', ylabel='x2')
    ax1.set_xlim([min(train_x[:, 0]) - 1, max(train_x[:, 0]) + 1])
    ax1.set_ylim([min(train_x[:, 1]) - 1, max(train_x[:, 1]) + 1])

    ax2.set_title("Loss graph")
    plt.setp(ax2, xlabel='#Epochs (Log scale)', ylabel='Loss')
    ax2.set_xlim([1 , epoch * 1.1])
    ax2.set_xscale('log')
    ax2.set_ylim([-0.1, max_loss * 1.1])

    if(len(loss) > 0):
        ax1.plot(X[:, 0][y_pred == 0], X[:, 1][y_pred == 0], "b.")
        ax1.plot(X[:, 0][y_pred == 1], X[:, 1][y_pred == 1], "m.")
        ax1.plot(X[:, 0][y_pred == 2], X[:, 1][y_pred == 2], "b.")
        ax1.set_title('Multi Class Classification with ' + str(train_accuracy) + '% accuracy after ' + str(f'{len(loss):,}') + ' epochs')

        ax2.plot(loss)
        ax2.set_title("Loss after " + str(f'{len(loss):,}') + " epochs is " + str("{:.6f}".format(loss[-1])))

    directory = "images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'images/{len(loss)}.png'
    for i in range(10):
        filenames.append(filename)
    # save frame
    plt.savefig(filename)
    plt.close()


epoch = 200000
learning_rate = 0.01
loss_log =[]
saved_epoch_no = 0
for epoch_no in range(epoch):
    loss = 0
    m = len(train_y)
    h = np.dot(train_x, w) + b
    yhat = sigmoid(h)

    loss = (-1 / m) * np.sum((train_y * np.log(yhat) + (1 - train_y) * np.log(1 - yhat)))

    dldw = (1 / m) * np.sum((yhat - train_y) * train_x, axis=0)
    dldw = np.array(dldw).reshape(-1,1)
    dldb = (1 / m) * np.sum((yhat - train_y))

    w -= learning_rate * dldw
    b -= learning_rate * dldb

    if epoch_no==0: plot_separation_line_and_loss(train_x, train_y, w, b, loss_log, epoch, loss)
    loss_log.append(loss)

    if ((epoch_no == 2) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        y_pred = predict(train_x, w, b)
        print("epoch_no: ", (epoch_no + 1), "\tloss:", loss, "\taccuracy:", accuracy(train_y, y_pred))
        if (epoch_no >= 2): plot_separation_line_and_loss(train_x, train_y, w, b, loss_log, epoch, max(loss_log))
        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))

create_gif(filenames, 'logistic_regression_linear_binary_classification.gif')