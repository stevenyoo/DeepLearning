from __future__ import print_function
import numpy as np
np.random.seed(1337)  

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from six.moves import cPickle

import deepctxt_util
from deepctxt_util import DCTokenizer

maxlen = 10 # cut texts after this number of words (among top max_features most common words)
batch_size = 100
epoch = 3
input_filename = 'D:/public/EQnA/QueryIntentClassifier/Inputs/test.eqna.txt'
output_basefilename = './data/test.eqna'
model_json_filename= './eqna_model_lstm_glove_100b.json'
model_weights_filename = './eqna_model_lstm_glove_100b.h5'

print('Loading tokenizer')
tokenizer = DCTokenizer()
tokenizer.load('./ner_dnn/glove.6B.100d.txt')

print('Loading data... (Test)')
(X2, y_test) = deepctxt_util.load_raw_data_x_y(path=input_filename)

print('Converting data... (Test)')
X_test = tokenizer.texts_to_sequences(X2, maxlen)
print(len(X_test), 'y_test sequences')

Y_test = np_utils.to_categorical(y_test)
print('Y_test shape:', Y_test.shape)

print("Padding sequences...")
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_test shape:', X_test.shape)

print('Loading model...')
with open(model_json_filename, 'r') as f:
	model_string = f.read()
	model = model_from_json(model_string)
	model.load_weights(model_weights_filename)

	score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)
	print('Test score:', score)
	print('Test accuracy:', acc)

	deepctxt_util.evaluate(model, X2, X_test, y_test, output_basefilename)

# print('Predict LSTM model...')
# Y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

# # write P/R/F1 per threshold
# from sklearn.metrics import precision_recall_curve
# outfile_pr = open('./data/test.do.pr.txt', 'w')
# precision, recall, threshold = precision_recall_curve(y_test, Y_pred[:,1])
# print('precission\trecall\tf1\tthreshold', file=outfile_pr)
# for i in range(len(precision)):
    # if i < len(precision) and i < len(recall) and i < len(threshold):
        # f1_score = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        # print(str(precision[i]) + '\t' + str(recall[i]) + '\t' + str(f1_score) + '\t' + str(threshold[i]), file=outfile_pr)
# outfile_pr.close()

# # write the prediction per query 
# outfile_prediction = open('./data/test.do.predicted.txt', 'w')
# print('Query\tTrueLabel\tPredicted', file=outfile_prediction)
# for i in range(len(X2)):
    # print(X2[i] + '\t' + str(y_test[i]) + '\t' + str(Y_pred[i,1]), file=outfile_prediction)
# outfile_prediction.close()