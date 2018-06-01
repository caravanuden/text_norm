# utils for LSTM encoder/decoder text normalization model
# mostly for data preprocessing -
# adding context window, padding to a fixed length vec,
# converting tokens to ints and sequences to np arrays
# CVU 2018
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding
from keras.layers.recurrent import LSTM
import numpy as np
import os

# replace word/character tokens with respective integers from vocabulary
def index(token_list, vocab):
    for token in token_list:
        for num in range(0, len(token)):
            if token[num] in vocab:
                token[num] = vocab[token[num]]
            else:
                token[num] = vocab['UNK']
    return token_list

# add context window to input sequences
def add_context_window(df, append_size, pad, vocab):

    df.extend([pad for i in range(append_size)])
    df.insert(0,pad) #Do in one line
    df.insert(0,pad)
    df.insert(0,pad)

    neo_data = []
    for i in range(append_size, (len(df) - append_size)):
        row = []
        row = [vocab['<norm>']] + df[i] + [vocab['<norm>']]
        row = ([item for sublist in df[(i - append_size):i] for item in sublist]
        + row
        + [item for sublist in df[(i + 1):(i + 1 + append_size)] for item in sublist])
        neo_data.append(row)
    return neo_data

# pad input and target sequences to create fixed length sequences
def padding_batchwise(data, max_len):
    i_end = 0
    for i in range(0, len(data),10000):
        if i + 10000 >= len(data):
            i_end = len(data)
        else:
            i_end = i + 10000
        data[i:i_end] = pad_sequences(data[i:i_end], maxlen = max_len, dtype='int32')
    return data

# convert 2D target sequences to 3D suitable for LSTM model
def sequences(data, seq_len, vocab):
    sequences = np.zeros((len(data), seq_len, len(vocab)))
    # Vectorizing each element in each sequence
    for i, sentence in enumerate(data):
        for j, word in enumerate(sentence):
                sequences[i, j, word] = 1.
    return sequences

# convert test data into array format, batch-wise
def array_batchwise(data, seq_len):
    i_end = 0
    array = np.empty((0,seq_len), dtype = int)
    for i in range(0, len(data),10000):
        if i + 10000 >= len(data):
            i_end = len(data)
        else:
            i_end = i + 10000
        arr = np.asarray(data[i:i_end], dtype = int)
        array = np.concatenate((array,arr), axis = 0)
    return array
