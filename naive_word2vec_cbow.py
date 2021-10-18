'''
Author: Gyan Mittal
Corresponding Document:
Brief about word2vec: A team of Google researchers lead by Tomas Mikolov developed, patented, and published Word2vec in two publications in 2013.
For learning word embeddings from raw text, Word2Vec is a computationally efficient predictive model.
Word2Vec methodology is used to calculate Word Embedding based on Neural Network/ iterative.
Word2Vec methodology have two model architectures: the Continuous Bag-of-Words (CBOW) model and the Skip-Gram model.
About Code: This code demonstrates the basic concept of calculating the word embeddings
using word2vec methodology using CBOW model.
'''

from collections import Counter
import itertools
import numpy as np
import re
import matplotlib.pyplot as plt
from util import *

# Clean the text after converting it to lower case
def naive_clean_text(text):
    text = text.strip().lower() #Convert to lower case
    text = re.sub(r"[^A-Za-z0-9]", " ", text) #replace all the characters with space except mentioned here
    return text

def prepare_training_data(corpus_sentences):
    window_size = 1
    split_corpus_sentences_words = [naive_clean_text(sentence).split() for sentence in corpus_sentences]

    center_word_train_y = []
    context_words_train_X = []

    word_counts = Counter(itertools.chain(*split_corpus_sentences_words))

    vocab_word_index = {x: i for i, x in enumerate(word_counts)}
    reverse_vocab_word_index = {value: key for key, value in vocab_word_index.items()}
    vocab_size = len(vocab_word_index)

    for sentence in split_corpus_sentences_words:
        for i in range(len(sentence)):
            center_word = [0 for x in range(vocab_size)]
            center_word[vocab_word_index[sentence[i]]] = 1
            #context = [0 for x in range(vocab_size)]

            for j in range(i - window_size, i + window_size+1):
                context = [0 for x in range(vocab_size)]
                if i != j and j >= 0 and j < len(sentence):
                    context[vocab_word_index[sentence[j]]] = 1
                    center_word_train_y.append(center_word)
                    context_words_train_X.append(context)

    return np.array(context_words_train_X), np.array(center_word_train_y), vocab_word_index

def initiate_weights(input_size, hidden_layer_size, output_size):
    np.random.seed(100)
    W1 = np.random.randn(input_size, hidden_layer_size)
    W2 = np.random.randn(hidden_layer_size, output_size)
    return W1, W2

def predict(X, W1, W2):
    Z1 = X.dot(W1)
    Z2 = Z1.dot(W2)
    y_pred = naive_softmax(Z2)
    y_pred = np.array(np.argmax(y_pred, axis=1))
    return y_pred

def back_propagation(train_X, train_y_one_hot_vector, yhat, Z1, W1, W2, learning_rate):

    #dl_dyhat = np.divide(train_y_one_hot_vector, pred_train_y)
    dl_dz2 = yhat - train_y_one_hot_vector
    dl_dw2 = Z1.T.dot(dl_dz2)

    dl_dz1 = dl_dz2.dot(W2.T)
    dl_dw1 = train_X.T.dot(dl_dz1)

    # update the weights
    W1 -= learning_rate * dl_dw1
    W2 -= learning_rate * dl_dw2

    return W1, W2

def forward_propagation(train_X, W1, W2):
    Z1 = train_X.dot(W1)
    Z2 = Z1.dot(W2)
    yhat = naive_softmax(Z2)
    return Z1, yhat

def plot_embeddings_and_loss(W, vocab_word_index, loss_log, epoch, max_loss, img_files=[]):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(10, 5))

    ax1.set_title('Word Embeddings in 2-d space for the given example')
    plt.setp(ax1, xlabel='Embedding dimension - 1', ylabel='Embedding dimension - 2')
    #ax1.set_xlim([min(W[:, 0]) - 1, max(W[:, 0]) + 1])
    #ax1.set_ylim([min(W[:, 1]) - 1, max(W[:, 1]) + 1])
    ax1.set_xlim([-3, 3.5])
    ax1.set_ylim([-3.5, 3])

    for word, i in vocab_word_index.items():
        x_coord = W[i][0]
        y_coord = W[i][1]
        #print(word, ":\t[", x_coord, ",", y_coord, "]")
        ax1.plot(x_coord, y_coord, "cD", markersize=5)
        #ax1.text(word, (x_coord, y_coord))
        ax1.text(x_coord, y_coord, word, fontsize=10)

    ax2.set_title("Loss graph")
    plt.setp(ax2, xlabel='#Epochs (Log scale)', ylabel='Loss')
    ax2.set_xlim([1 , epoch * 1.1])
    ax2.set_xscale('log')
    ax2.set_ylim([0, max_loss * 1.1])

    if(len(loss_log) > 0):
        ax2.plot(1, max(loss_log), "bD")
        ax2.plot(loss_log, "b")
        ax2.set_title("Loss is " + r"$\bf{" + str("{:.6f}".format(loss_log[-1])) + "}$" + " after " + r"$\bf{" + str(f'{len(loss_log) - 1:,}') + "}$" + " epochs")

    directory = "images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'images/{len(loss_log)}.png'
    for i in range(13):
        img_files.append(filename)
    # save frame
    plt.savefig(filename)
    plt.close()
    return img_files



corpus_sentences = ["I love playing Football", "I love playing Cricket", "I love playing sports"]

train_X, train_y_one_hot_vector, vocab_word_index = prepare_training_data(corpus_sentences)

# Network with one hidden layer
input_size = len(train_X[0]) # Number of input features
hidden_layer_size = 2 # Design choice [Embedding Dimension]
output_size = len(vocab_word_index) # Number of classes

W1, W2 = initiate_weights(input_size, hidden_layer_size, output_size)

epoch = 200000
learning_rate = 0.0001
loss_log =[]

saved_epoch_no = 0
for epoch_no in range(epoch):
    loss = 0

    Z1, yhat = forward_propagation(train_X, W1, W2)
    loss = cross_entropy_loss(train_y_one_hot_vector, yhat)
    if (epoch_no==0):
        image_files = plot_embeddings_and_loss(W1, vocab_word_index, loss_log, epoch, loss)
        loss_log.append(loss)

    loss_log.append(loss)

    W1, W2= back_propagation(train_X, train_y_one_hot_vector, yhat, Z1, W1, W2, learning_rate)

    if ((epoch_no == 1) or np.ceil(np.log10(epoch_no + 2)) > saved_epoch_no or (epoch_no + 1) == epoch):
        print("epoch_no: ", (epoch_no + 1), "\tloss_log:", loss)
        image_files = plot_embeddings_and_loss(W1, vocab_word_index, loss_log, epoch, max(loss_log), image_files)

        saved_epoch_no = np.ceil(np.log10(epoch_no + 2))


create_gif(image_files, 'images/word2vec_cbow.gif')