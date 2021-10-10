'''
Author: Gyan Mittal
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

# Training data
#train_y = w * train_x + b
train_x = np.array([3,4,5,6,7,8,9])
train_y = np.array([7.2,9.1,10.8,12.5,13.5,17.4,18])

# Initialize w and b with some random number
np.random.seed(100)
w = np.random.rand()
b = np.random.rand()

# Plotting
filenames = []
def plot_separation_line_and_loss(train_x, train_y, w, b, loss, epoch, max_loss):

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(18, 9))
    ax1.plot(train_x, train_y, "cD")
    ax1.set_title('Linear Regression training data')
    plt.setp(ax1, xlabel='x', ylabel='y')
    ax1.set_xlim([0, max(train_x) + 1])
    ax1.set_ylim([0, max(train_y) + 1])

    ax2.set_title("Loss graph")
    plt.setp(ax2, xlabel='#Epochs (Log scale)', ylabel='Loss')
    ax2.set_xlim([1 , epoch * 1.1])
    ax2.set_xscale('log')
    ax2.set_ylim([-.2, max_loss * 1.1])

    if(len(loss) > 0):
        x = np.array([0, max(train_x) + 1])
        y = w * x + b
        ax1.plot(x, y)
        ax1.set_title('Linear Regression' + ' after ' + str(f'{len(loss):,}') + ' epochs')

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
    #plt.show()
    plt.close()

def create_gif(filenames):
    # build gif
    with imageio.get_writer('linear_regression.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove files
    for filename in set(filenames):
        os.remove(filename)


epoch = 200000
learning_rate = 0.00001
loss_log =[]

saved_epoch_no = 0
for epoch_no in range(epoch):
    yhat = w * train_x + b

    no_samples = len(train_x)
    loss = (1/no_samples) * np.sum(np.square(yhat - train_y))
    if (epoch_no == 0): plot_separation_line_and_loss(train_x, train_y, w, b, loss_log, epoch, loss)
    loss_log.append(loss)
    dldb = 2 * np.sum(yhat - train_y)
    dldw = 2 * np.sum((yhat - train_y) * train_x)
    w -= learning_rate * dldw
    b -= learning_rate * dldb

    if ((epoch_no == 2) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        print("epoch_no: ", (epoch_no + 1), "\tloss:", loss)
        if (epoch_no >= 2): plot_separation_line_and_loss(train_x, train_y, w, b, loss_log, epoch, max(loss_log))
        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))

print("w: ", w)
print("b: ", b)
create_gif(filenames)


