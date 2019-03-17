import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]



def load_newsdata_and_labels():
    """
    Read newsdata, return list of documents, each line in list is one document as string. 
    And list of labels, each line in list is one-hot-encoded class
    """
    # read newsdata which is pickled
    import pickle
    def read_pickle_one_by_one(pickle_file):
        with open(pickle_file, "rb") as t_in:
                while True:
                    try:
                        yield pickle.load(t_in)
                    except EOFError:
                        break

    #sentnos = [s for s in read_pickle_one_by_one("sentnos.pkl")] # sentence numbers
    labels  = [l for l in read_pickle_one_by_one("data_own/labels.pkl")]
    #focuses = [f for f in read_pickle_one_by_one("focuses.pkl")]
    texts   = [t for t in read_pickle_one_by_one("data_own/texts.pkl")]

    # assert  == len(labels) == len(texts) # == len(sentnos) == len(focuses)

    #print("longest text")
    #print(max(len(t) for t in texts))

    #print(sentnos[23])
    #print(texts[23])
    #print(focuses[23])
    #print(labels[23])

    # import copy
    # if need real copies, not just new pointers
    # new_texts = copy.deepcopy(texts)
    
    # empty list, same lenght as texts
    new_texts = [None] * len(texts)
    
    # go through list and for each document in list, join list of words to a string
    for documentnr, value in enumerate(texts):
        #print(document, value)
        new_texts[documentnr] = ' '.join(value)

    # labels are 5-6 classes. turn them into 1-hot-encoded. 6 classes mentioned in paper, only 5 present in data.
    new_labels = np.zeros((len(labels),5))
    for labelnr, value in enumerate(labels):
        if value[0]==1:
            new_labels[labelnr][0]=1  #one hot to true

        elif value[0]==0.7:
            new_labels[labelnr][1]=1  

        elif value[0]==0.5:
            new_labels[labelnr][2]=1  

        elif value[1]==0.7:
            new_labels[labelnr][3]=1  

        elif value[0]==0:
            new_labels[labelnr][4]=1  

    x_text = new_texts
    y = new_labels
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# loading of word2vec and glove from
# https://github.com/cahya-wirawan/cnn-text-classification-tf/blob/master/data_helpers.py
def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors

def load_newsdata_with_focus():
    """
    Read newsdata, return list of documents, each line in list is one document as string. 
    And list of labels, each line in list is one-hot-encoded class.
    Also return focuses as position(s) lists of target companies.
    """
    # read newsdata which is pickled
    import pickle
    def read_pickle_one_by_one(pickle_file):
        with open(pickle_file, "rb") as t_in:
                while True:
                    try:
                        yield pickle.load(t_in)
                    except EOFError:
                        break

    labels  = [l for l in read_pickle_one_by_one("data_own/labels.pkl")]
    focuses = [f for f in read_pickle_one_by_one("data_own/focuses.pkl")]
    texts   = [t for t in read_pickle_one_by_one("data_own/texts.pkl")]

    # empty list, same lenght as texts
    x_text = [None] * len(texts)
    
    # go through list and for each document in list, join list of words to a string
    for documentnr, value in enumerate(texts):
        x_text[documentnr] = ' '.join(value)

    # labels are 5-6 classes. turn them into 1-hot-encoded. 6 classes mentioned in paper, only 5 present in data.
    y = [vectoriseLabels(x,5) for x in labels]

    return [x_text, focuses, y]

def vectoriseLabels(label, length):
    import numpy as np
    ret = np.zeros(length)
    # labels are 5-6 classes. turn them into 1-hot-encoded. 6 classes mentioned in paper, only 5 present in data.
    # label is a list [POSITIVE_PROBABILITY, NEGATIVE_PROBABILITY]
    dictionary = {1:0, 0.3:1, 0.5:2, 0.7:3, 0:4}
    ret[dictionary[round(label[0],1)]] = 1
    return ret

def make_attention_matrix(focus, shape, distribute = False):
    # focus -- one number (position) for each sentence
    # attention -- matrix of zeros, 1 on focus positions
    attention = np.zeros(shape)
    for s in range(len(focus)):
        # loop by sentence
        if focus[s] is not None:  # None - no focus (since we have very dirty data)
            # focus[s] means focus(es) for the sentence
            try:
                if isinstance(focus[s], list):
                    focus_list = focus[s]
                else:
                    # inconsistencies in Nsents and SbyS, need to check it, should be always list
                    focus_list = [focus[s]]
                if None in focus_list:
                    continue
                if distribute:
                    for i in range(len(attention[s])):
                            attention[s][i][0] = compute_attention_distance(i, focus[s])
                else:
                    for f in focus[s]:
                        attention[s][f][0] = 1
            except IndexError:
                # focus outside sentences - (may happen e.g. in testing if model trained with shorter sentences than in test) or if two long sentences are paired together
                if len(attention[s]) <= f:
                    continue
                else:
                    # something else, some bug
                    exit(1)
    return attention

def compute_attention_distance(current_position, list_of_focuses):
    try:
        return max([1.0 / (1.0 + abs(current_position - focus_position)) for focus_position in list_of_focuses])
    except:
        return 0