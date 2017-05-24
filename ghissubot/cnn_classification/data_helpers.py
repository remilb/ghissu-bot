import numpy as np
import re
import itertools
from collections import Counter
import os
from statsmodels.tools import categorical
import pandas as pd


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

def load_swbd_data(sequence_length, data_dir= os.getcwd() + "/data/switchboard"):

    prev_dir = os.getcwd()
    os.chdir(data_dir)
    x_text = list(open(data_dir + "/swbd_utterance.csv", "r").readlines())
    x_text=  [s.strip() for s in x_text]
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [list(x.split()) for x in x_text]
    #x_text = np.asanyarray(x_text)

    row = ["ENDPADDING"] * sequence_length
    x_text_temp = []
    for i in range(len(x_text)):
        x_text_temp.append(row[:])


    for i in range(len(x_text)):
        for j in range(min(sequence_length, len(x_text[i]))):
            x_text_temp[i][j] = x_text[i][j]

    for i in range(5):
        print("lists")
        print(x_text[i])
        print(x_text_temp[i])

    x_test_temp = np.asanyarray(x_text_temp)
    print(x_test_temp.shape)
    y = pd.read_csv(data_dir + "/swbd_act.csv")
    y = list(open(data_dir + "/swbd_act.csv", "r").readlines())
    a = np.array([s.strip() for s in y]) # ["fx" , "qa" ]
    y = categorical(a, drop=True) # 3 = [0 0 0 1 0 0 0 0 0 ....]
    #y = y.argmax(axis=1)  # 3
    '''
    from scikits.statsmodels.tools import categorical

    In [61]: a = np.array( ['a', 'b', 'c', 'a', 'b', 'c'])
    
    In [62]: b = categorical(a, drop=True)
    
    In [63]: b.argmax(1)
    Out[63]: array([0, 1, 2, 0, 1, 2])
    '''
    #return [x_text, y]
    return [x_text_temp, y]


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
