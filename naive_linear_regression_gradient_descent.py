'''
Author: Gyan Mittal
Corresponding Document: https://gyan-mittal.com/nlp-ai-ml/nlp-word-encoding-by-one-hot-encoding/
Brief about One–Hot–Encoding:
One of the simplest forms of word encoding to represent the word in NLP is One–Hot–Vector–Encoding.
It requires very little computing power to convert text data into one-hot encoding data, and it’s easy to implement.
One–Hot–Encoding has the advantage over Label/ Integer encoding, that the result is binary rather than ordinal, it does not suffer from undesirable bias.
About Code: This code demonstrates the concept of One–Hot–Encoding with two simple example corpus
'''
import numpy as np
print("naive regression")

# Training data
#y = wx + c
train_x = np.array([3,5,6,8,9])
train_y = np.array([7,11,13,17,19])

# Initialize w and c with some random number
w = 5
c = 10

# Prediceted y
yhat = w * train_x + c

epoch = 100000
learning_rate = 0.001
for epoch_no in range(epoch):
    loss = 0
    for i, _ in enumerate(yhat):
        loss += (yhat[i] - train_y[i]) ** 2

    if (epoch_no == 0 or (epoch_no+1)%10000 == 0):
        print("epoch_no: ", epoch_no, "\tloss:", loss)
    dldc = 0
    dldw = 0

    #Gradient Descent
    for i, _ in enumerate(yhat):
        dldc += 2 * (yhat[i] - train_y[i])
        dldw += 2 * (yhat[i] - train_y[i]) * train_x[i]
    dldc /= len(yhat)
    dldw /= len(yhat)
    w -= learning_rate * dldw
    c -= learning_rate * dldc
    yhat = w * train_x + c

print("w: ", w)
print("c: ", c)