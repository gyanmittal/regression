'''
Author: Gyan Mittal
Corresponding Document:
Brief about linear regression gradient descent:

'''
import numpy as np

# Training data
#train_y = wx + b
train_x = np.array([3,4,5,6,7,8,9])
train_y = np.array([7.2,9.1,10.8,12.5,13.5,17.4,18])

# Initialize w and b with some random number
w = 5
b = 10

# Prediceted train_y
yhat = w * train_x + b

epoch = 200000
learning_rate = 0.001
for epoch_no in range(epoch):
    loss = 0
    for i, _ in enumerate(yhat):
        loss += (yhat[i] - train_y[i]) ** 2 / len(yhat)

    if (epoch_no == 0 or (epoch_no+1)%10000 == 0):
        print("epoch_no: ", epoch_no, "\tloss:", loss)
    dldb = 0
    dldw = 0

    #Batch Gradient Descent
    for i, _ in enumerate(yhat):
        dldb += 2 * (yhat[i] - train_y[i])
        dldw += 2 * (yhat[i] - train_y[i]) * train_x[i]
    dldb /= len(yhat)
    dldw /= len(yhat)
    w -= learning_rate * dldw
    b -= learning_rate * dldb
    yhat = w * train_x + b

print("w: ", w)
print("b: ", b)