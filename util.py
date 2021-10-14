import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def naive_softmax(x):
    return np.exp(x)/np.exp(x).sum(axis=1, keepdims=True)

def accuracy(y, y_pred):
    acc = int(sum(y == y_pred) / len(y) * 100)
    return acc

# Plotting linear regression
def plot_linear_regression_line_and_loss(train_x, train_y, W, b, loss_log, epoch, max_loss, img_files=[]):

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(18, 9))
    ax1.plot(train_x, train_y, "cD", markersize=7)
    ax1.set_title('Linear Regression training data')
    plt.setp(ax1, xlabel='x', ylabel='y')
    ax1.set_xlim([0, max(train_x) + 1])
    ax1.set_ylim([0, max(train_y) + 1])

    ax2.set_title("Loss graph")
    plt.setp(ax2, xlabel='#Epochs (Log scale)', ylabel='Loss')
    ax2.set_xlim([1, epoch * 1.1])
    ax2.set_xscale('log')
    ax2.set_ylim([-1, max_loss * 1.1])

    if(len(loss_log) > 0):
        x = np.array([0, max(train_x) + 1])
        y = W * x + b
        ax1.plot(x, y, "c")
        ax1.set_title('Linear Regression' + " after " + r"$\bf{" + str(f'{len(loss_log) - 1:,}') + "}$" + " epochs")
        ax2.plot(loss_log, "b")
        ax2.set_title("Loss is " + r"$\bf{" + str("{:.6f}".format(loss_log[-1])) + "}$" + " after " + r"$\bf{" + str(f'{len(loss_log) - 1:,}') + "}$" + " epochs")
    directory = "images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'images/{len(loss_log)}.png'
    for i in range(6):
        img_files.append(filename)
    # save frame
    plt.savefig(filename)
    plt.close()
    return img_files

# Plotting logistic regression
def plot_classification_separation_line_and_loss(train_X, actual_train_y, pred_train_y, test_data_X, pred_test_data_y, loss_log, epoch, max_loss, img_files=[]):

    train_accuracy = accuracy(actual_train_y, pred_train_y)
    # Plotting
    #fig = plt.figure(figsize=(9, 9))
    fig, (ax1, ax2) = plt.subplots(1,2, sharex=False, figsize=(18, 9))

    actual_train_y = actual_train_y.flatten()
    ax1.plot(train_X[:, 0][actual_train_y == 0], train_X[:, 1][actual_train_y == 0], "cD", markersize=7)
    ax1.plot(train_X[:, 0][actual_train_y == 1], train_X[:, 1][actual_train_y == 1], "mD", markersize=7)
    ax1.plot(train_X[:, 0][actual_train_y == 2], train_X[:, 1][actual_train_y == 2], "bD", markersize=7)

    ax1.set_title('Multi Class Classification training data')
    plt.setp(ax1, xlabel='x1', ylabel='x2')
    ax1.set_xlim([min(train_X[:, 0]) - 1, max(train_X[:, 0]) + 1])
    ax1.set_ylim([min(train_X[:, 1]) - 1, max(train_X[:, 1]) + 1])

    ax2.set_title("Loss graph")
    plt.setp(ax2, xlabel='#Epochs (Log scale)', ylabel='Loss')
    ax2.set_xlim([1 , epoch * 1.1])
    ax2.set_xscale('log')
    ax2.set_ylim([-0.1, max_loss * 1.1])

    if(len(loss_log) > 0):
        ax1.plot(test_data_X[:, 0][pred_test_data_y == 0], test_data_X[:, 1][pred_test_data_y == 0], "c.", markersize=2)
        ax1.plot(test_data_X[:, 0][pred_test_data_y == 1], test_data_X[:, 1][pred_test_data_y == 1], "m.", markersize=2)
        ax1.plot(test_data_X[:, 0][pred_test_data_y == 2], test_data_X[:, 1][pred_test_data_y == 2], "b.", markersize=2)
        ax1.set_title('Multi Class Classification with ' + r"$\bf{" + str(train_accuracy) + "}$" + '% accuracy after ' + r"$\bf{" + str(f'{len(loss_log) - 1:,}') + "}$" + ' epochs')

        ax2.plot(loss_log, "b")
        ax2.set_title("Loss is " + r"$\bf{" + str("{:.6f}".format(loss_log[-1])) + "}$" + " after " + r"$\bf{" + str(f'{len(loss_log) - 1:,}') + "}$" + " epochs")

    directory = "images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'images/{len(loss_log)}.png'
    for i in range(6):
        img_files.append(filename)
    # save frame
    plt.savefig(filename)
    plt.close()
    return img_files

def create_gif(input_image_filenames, output_gif_name):
    # build gif
    with imageio.get_writer(output_gif_name, mode='I') as writer:
        for filename in input_image_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove files
    for filename in set(input_image_filenames):
        os.remove(filename)