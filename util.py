import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def accuracy(y, y_pred):
    acc = int(sum(y == y_pred) / len(y) * 100)
    return acc

def create_gif(input_image_filenames, output_gif_name):
    # build gif
    with imageio.get_writer(output_gif_name, mode='I') as writer:
        for filename in input_image_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove files
    for filename in set(input_image_filenames):
        os.remove(filename)