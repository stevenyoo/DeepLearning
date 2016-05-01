# coding: utf-8
from __future__ import absolute_import

import string
import sys
import numpy as np
from six.moves import range
from six.moves import zip
from sklearn.metrics import precision_recall_curve

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

# write p/r/f1 per threshold
def evaluate(model, X_instance, X_test, y_test, base_filename):
	print('Predict classes...')
	y_class = model.predict_classes(X_test)

	outfile_pr = open(base_filename + '.pr.txt', 'w')
	precision, recall, threshold = precision_recall_curve(y_test, y_class)
	print('precission\trecall\tf1\tthreshold', file=outfile_pr)
	for i in range(len(precision)):
		if i < len(precision) and i < len(recall) and i < len(threshold):
			f1_score = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
			print(str(precision[i]) + '\t' + str(recall[i]) + '\t' + str(f1_score) + '\t' + str(threshold[i]), file=outfile_pr)
	outfile_pr.close()
	
	# write the prediction per query 
	outfile_prediction = open(base_filename + '.predicted.txt', 'w')
	print('query\ttruelabel\tpredicted', file=outfile_prediction)
	for i in range(len(X_instance)):
		print(X_instance[i] + '\t' + str(y_test[i]) + '\t' + str(y_class[i]), file=outfile_prediction)
	outfile_prediction.close()

def load_raw_data_x_y(path="./raw_data.tsv", y_shift=0):
    X = []
    Y = []
    f = open(path, "r", encoding='utf-8')
    for l in f:
        line = l.replace("\n", "")
        fields = line.split('\t')
        if len(fields) != 2:
            continue
        x = fields[0]
        y = int(fields[1].strip()) + y_shift
        if len(x) <= 0:
            continue
        X.append(x)
        Y.append(y)
    f.close()
    return (X, Y)

def load_raw_data_x1_x2_y(path="./raw_data.tsv", y_shift=0):
    X1 = []
    X2 = []
    Y = []
    f = open(path, "r", encoding='utf-8')
    for l in f:
        line = l.replace("\n", "")
        fields = line.split('\t')
        if len(fields) != 3:
            continue
        x1 = fields[0]
        x2 = fields[1]
        y = int(fields[2].strip()) + y_shift
        if len(x1) <= 0:
            continue
        if len(x2) <= 0:
            continue
        X1.append(x1)
        X2.append(x2)
        Y.append(y)
    f.close()
    return (X1,X2,Y)

def is_in_vocab(vocabs, t):
    terms = t.split(' ')
    for term in terms:
        if not term in vocabs:
            return False
    return True

def load_raw_data_termx(path="./raw_data.tsv", y_shift=0, seed=1337, vocabs=None, add_reverse=False):
    X = []
    X2 = []
    Y = []
    f = open(path, "r", encoding='utf-8')
    for l in f:
        line = l.replace("\n", "")
        fields = line.split('\t')
        if len(fields) != 4:
            continue
        x = fields[0]
        src = fields[1]
        tgt = fields[2]
        y = int(fields[3].strip()) + y_shift
        if len(x) <= 0:
            continue
        x2 = x.replace(src, tgt)
        if x == x2:
            continue

        if vocabs != None:
            if (not is_in_vocab(vocabs, src)) or (not is_in_vocab(vocabs, tgt)):
                continue

        X.append(x)
        X2.append(x2)
        Y.append(y)

        if add_reverse:
            X.append(x2)
            X2.append(x)
            Y.append(y)

    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(X2)
    np.random.seed(seed)
    np.random.shuffle(Y)

    return (X, X2, Y)

def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f

def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split*len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]

class DCTokenizer(object):
    def __init__(self, nb_words=None, filters=base_filter(),
                 lower=True, split=' '):

        # reserve inex 0..9 for special purpose
        # 0 -> NOT_USED
        # 1 -> OOV
        # 2 -> BEGIN 
        # 3 -> reserve
        # 4 -> reserve
        # 5 -> reserve
        # 6 -> reserve
        # 7 -> reserve
        # 8 -> reserve
        # 9 -> reserve

        self.vocab_dim = -1
        self.n_symbols = -1
        self.word2index = {}
        self.embedding_weights = None
        self.index_oov = -1
        self.index_begin = -1
        self.nb_words = nb_words
        self.filters = filters
        self.lower = lower
        self.split = split

    def load(self, filename):
        self.word2index = {}

        self.word2index["_NOT_USED_"] = 0
        self.word2index["_OOV_"] = 1
        self.word2index["_BEGIN_"] = 2
        self.word2index["_RESV3_"] = 3
        self.word2index["_RESV4_"] = 4
        self.word2index["_RESV5_"] = 5
        self.word2index["_RESV6_"] = 6
        self.word2index["_RESV7_"] = 7
        self.word2index["_RESV8_"] = 8
        self.word2index["_RESV9_"] = 9

        self.index_oov = self.word2index["_OOV_"]
        self.index_begin = self.word2index["_BEGIN_"]

        self.vocab_dim = -1

        word_count = 0
        with open(filename, 'r', encoding='utf-8') as f_in:
            for l in f_in:
                if self.vocab_dim < 0:
                    fields = l.replace("\n","").split(' ')
                    weights = np.fromstring(" ".join(fields[1:]), dtype=float, sep=' ')
                    self.vocab_dim = len(weights)
                word_count += 1
        self.n_symbols = len(self.word2index) + word_count
        self.embedding_weights = np.zeros((self.n_symbols, self.vocab_dim))

        index = len(self.word2index)
        with open(filename, 'r', encoding='utf-8') as f_in:
            for l in f_in:
                fields = l.replace("\n","").split(' ')
                word = fields[0]
                weights = np.fromstring(" ".join(fields[1:]), dtype=float, sep=' ')
                self.word2index[word] = index
                self.embedding_weights[index,:] = weights
                index += 1

        print("n_symbols=" + str(self.n_symbols))
        print("vocab_dim=" + str(self.vocab_dim))


    def texts_to_sequences(self, texts, maxlen):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Returns a list of sequences.
        '''
        res = []
        for vect in self.texts_to_sequences_generator(texts, maxlen):
            res.append(vect)
        return res

    def texts_to_sequences_generator(self, texts, maxlen):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Yields individual sequences.
        '''
        nb_words = self.nb_words
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word2index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        vect.append(self.index_oov)
                    else:
                        vect.append(i)
                else:
                    vect.append(self.index_oov)
                if maxlen > 0 and len(vect) >= maxlen:
                    break
            yield vect
