###
# import packages
###
import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

##
# Data Load
###
X_train = np.load('./X_train_rnn.npy', allow_pickle=True)
X_test = np.load('./X_test_rnn.npy', allow_pickle=True)
y_train = np.load('./y_train.npy')
y_test = np.load('./y_test.npy')


###
# Data preprocessing
###
X_tr = pd.DataFrame(X_train)[0].str.split(' ').tolist()
for i in range(len(X_tr)):
    for j in range(len(X_tr[i])):
        if len(X_tr[i][j]) > 7:
            X_tr[i][j] = X_tr[i][j][X_tr[i][j].find(':')+1:]
    while '' in X_tr[i]:
        X_tr[i].remove('')
    X_tr[i] = ' '.join(' '.join(X_tr[i]).split(':'))
        
X_te = pd.DataFrame(X_test)[0].str.split(' ').tolist()
for i in range(len(X_te)):
    for j in range(len(X_te[i])):
        if len(X_te[i][j]) > 7:
            X_te[i][j] = X_te[i][j][X_te[i][j].find(':')+1:]
    while '' in X_te[i]:
        X_te[i].remove('')
    X_te[i] = ' '.join(' '.join(X_te[i]).split(':'))

#sectence to vector
all_text = ' '.join([c for c in X_tr])

words = all_text.split()
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)
vocab_to_int = {w:i+2 for i, (w,c) in enumerate(sorted_words)}
vocab_to_int['<unk>']=0
vocab_to_int['<pad>']=1

X_tr_int = []
for one in X_tr:
    one_int = [vocab_to_int[w] for w in one.split()]
    X_tr_int.append(one_int)
    
X_te_int = []
for one in X_te:
    one_int = [vocab_to_int[w] for w in one.split() if w in vocab_to_int]
    X_te_int.append(one_int)

## Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
def pad_features(X_int, seq_length):
    features = np.zeros((len(X_int), seq_length), dtype = int)
    for i, one in enumerate(X_int):
        one_len = len(one)
        if one_len <= seq_length:
            zeroes = list(np.zeros(seq_length-one_len))
            new = zeroes+one
        elif one_len > seq_length:
            new = one[0:seq_length]
        features[i,:] = np.array(new)
    return features

seq_length = 300
indexed_X_tr = pad_features(X_tr_int, seq_length)
indexed_X_te = pad_features(X_te_int, seq_length)
indexed_y_tr = y_train
indexed_y_te = y_test

len(indexed_y_te)
train_data = TensorDataset(torch.from_numpy(indexed_X_tr), torch.from_numpy(indexed_y_tr))
test_data = TensorDataset(torch.from_numpy(indexed_X_te), torch.from_numpy(indexed_y_te))

batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

###
# Set the arguments
###
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(vocab_to_int) + 1   ########## Q) Why do we add + 1?
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 1


###
# GRU Classifier model define
###
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers):
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)     ######## padding_idx
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first = True)

        self.output_fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        self.batch_size = x.size(0)
        embeds = self.embedding(x).to(device)
        h0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(device)

        output, hidden = self.gru(embeds, h0)
        output = self.output_fc(output[:, -1, :])
        output = self.sig(output)
        return output


###
# Load the model
###
model = GRUClassifier(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.load_state_dict(torch.load('model_parameter_rnn.pt'))
model = model.to('cuda')

###
# Evaluate using AUROC and AUPRC
###
model.eval()
train_pred = []
train_prob = []
train_acc = []
train_label = []
for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    output = model(inputs)
    prob = output.squeeze()
    pred = (prob > 0.5).float()
    acc = torch.mean((pred == labels).float())
    train_pred.append(pred)
    train_acc.append(acc.item())
    train_prob += prob.tolist()
    train_label += labels.tolist()

model.eval()
test_pred = []
test_prob = []
test_acc = []
test_label = []
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    output = model(inputs)
    prob = output.squeeze()
    pred = (prob > 0.5).float()
    acc = torch.mean((pred == labels).float())
    test_pred.append(pred)
    test_prob += prob.tolist()
    test_acc.append(acc.item())
    test_label += labels

train_roc = round(roc_auc_score(y_train, train_prob), 4)
test_roc = round(roc_auc_score(y_test, test_prob), 4)
train_prc = round(average_precision_score(y_train, train_prob), 4)
test_prc = round(average_precision_score(y_test, test_prob), 4)

###
# Save the result of evaluation at './20214577_rnn.txt'
###
with open('./20214577_rnn.txt', 'w') as f:
    f.write(f'20214577\n{train_roc}\n{train_prc}\n{test_roc}\n{test_prc}\n')