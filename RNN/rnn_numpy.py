# https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
import numpy as np
import pandas as pd

# load text data

txt_data = "abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz "  # input data
num_chars = 27
col = pd.Series([char for char in txt_data])
char_vectors = pd.get_dummies(col)
# hyperparameters

iteration = 5000
sequence_length = 10
batch_size = round((len(txt_data) / sequence_length) + 0.5)  # = math.ceil
hidden_size = 100  # size of hidden layer of neurons.
learning_rate = 1e-1

# model parameters

W_xh = np.random.randn(hidden_size, num_chars) * 0.01  # weight input -> hidden.
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # weight hidden -> hidden
W_hy = np.random.randn(num_chars, hidden_size) * 0.01  # weight hidden -> output

b_h = np.zeros((hidden_size, 1))  # hidden bias
b_y = np.zeros((num_chars, 1))  # output bias

h_prev = np.zeros((hidden_size, 1))  # h_(t-1)
