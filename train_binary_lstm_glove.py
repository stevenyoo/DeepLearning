# coding: utf8
from __future__ import print_function
import numpy as np
np.random.seed(1337)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from sklearn.metrics import precision_recall_curve
import string
import deepctxt_util

maxlen = 10 # cut texts after this number of words (among top max_features most common words)
batch_size = 100
epoch = 3
input_train_filename = 'D:/public/EQnA/QueryIntentClassifier/Inputs/train.txt'
input_test_filename = 'D:/public/EQnA/QueryIntentClassifier/Inputs/test.do.txt'
output_basefilename = './data/test.do'
model_json_filename= './eqna_model_lstm_glove_100b.json'
model_weights_filename = './eqna_model_lstm_glove_100b.h5'

tokenizer = deepctxt_util.DCTokenizer()
print('Loading tokenizer')
tokenizer.load('./ner_dnn/glove.6B.100d.txt')

print('Loading data...')
(X1, y_train) = deepctxt_util.load_raw_data_x_y(path=input_train_filename)
(X2, y_test) = deepctxt_util.load_raw_data_x_y(path=input_test_filename)

print('Converting data...')
X_train = tokenizer.texts_to_sequences(X1, maxlen)
X_test = tokenizer.texts_to_sequences(X2, maxlen)

print(len(X_train), 'y_train sequences')
print(len(X_test), 'y_test sequences')

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print("Pad sequences...")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=tokenizer.n_symbols, output_dim=tokenizer.vocab_dim, input_length=maxlen, weights=[tokenizer.embedding_weights]))
model.add(LSTM(128))  
model.add(Dropout(0.5))
model.add(Dense(np.max(y_train)+1))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam')

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch, show_accuracy=True)

score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)

json_model_string = model.to_json()
with open(model_json_filename, "w") as f:
    f.write(json_model_string)
model.save_weights(model_weights_filename)

deepctxt_util.evaluate(model, X2, X_test, y_test, output_basefilename)