'''
Author: Gyan Mittal
Corresponding Document:
Brief about linear regression gradient descent:

'''
import numpy as np
from util import *

#Training data
#train_y = W * train_X + b
train_X = np.array([3, 4, 5, 6, 7, 8, 9])
train_y = np.array([7.2,9.1,10.8,12.5,13.5,17.4,18])

# Initialize W and b with some random number
np.random.seed(100)
W = np.random.rand()
b = np.random.rand()

epoch = 20000
learning_rate = 0.001
loss_log =[]

saved_epoch_no = 0
for epoch_no in range(epoch):
    loss  = 0
    dldb = 0
    dldw = 0
    #Batch Gradient Descent
    yhat = W * train_X + b
    for i, _ in enumerate(train_y):
        dldb += 2 * (yhat[i] - train_y[i])
        dldw += 2 * (yhat[i] - train_y[i]) * train_X[i]
        loss += (yhat[i] - train_y[i]) ** 2
    loss = (1/len(train_y)) * loss
    dldb /= len(yhat)
    dldw /= len(yhat)
    W -= learning_rate * dldw
    b -= learning_rate * dldb

    if (epoch_no == 0):
        image_files = plot_linear_regression_line_and_loss(train_X, train_y, W, b, loss_log, epoch, loss)
        loss_log.append(loss)
    loss_log.append(loss)

    if ((epoch_no == 1) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        print("epoch_no: ", (epoch_no + 1), "\tloss_log:", loss)
        if (epoch_no >= 1):
            image_files = plot_linear_regression_line_and_loss(train_X, train_y, W, b, loss_log, epoch, max(loss_log), image_files)
        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))

print("W: ", W)
print("b: ", b)

create_gif(image_files, 'images/linear_regression_gradient_descent.gif')