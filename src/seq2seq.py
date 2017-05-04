import  pandas as pd
import numpy as np
import os
import tensorflow as tf
import copy
from tensorflow.contrib import learn
import _pickle as pickle

def read_from_csv():

    csv_fname = "/Users/shubhi/Public/CMPS296/friends.csv" #replace with local file loc
    df = pd.DataFrame.from_csv(csv_fname)
    df = df.dropna(subset = ['utterance'])
    print(len(df))
    X = list(df['utterance'])
    
    Y = copy.deepcopy(X)
    X = X[:-1]
    Y = Y[1:]

    #print(len(X))
    #print(len(Y))
    return (X, Y)

def loadGloVe(filename):

#    pwd = os.getcwd()
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab, embd


def main():

    sess = tf.Session()
    (X,Y) = read_from_csv()
    max_sentence_length = 20

    glove_filename = '/Users/shubhi/Public/CMPS296/glove.6B/glove.6B.50d.txt' #replace with local file loc
    vocab, embd = loadGloVe(glove_filename)
    
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)

    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    X = np.array(list(vocab_processor.transform(X)))
    
    print (X[0])
    
    with tf.Session() as sess:
        
    #    TF variable (placeholder)
        W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        embedding_init = W.assign(embedding_placeholder)
        
        req_embedded = tf.nn.embedding_lookup(W, X)
        
#        Call the session
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

        vectors = req_embedded.eval()
        print (vectors[0])
        pickle.dump(vectors, open("vectorized_input", "wb"))        

main()
