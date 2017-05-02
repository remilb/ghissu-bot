import  pandas as pd
import numpy as np
import os
import tensorflow as tf
import  copy
from tensorflow.contrib import learn

def read_from_csv():

    df = pd.DataFrame.from_csv("/home/sharath/programming/friends.csv")
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

    filename = '/home/sharath/programming/glove.6B.100d.txt'

    vocab, embd = loadGloVe(filename)
    
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    print embedding[0], vocab[0]
    
    
    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    X = np.array(list(vocab_processor.transform(X)))
    
    print X[0]
    
    
#    TF variable (placeholder)
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    tf.nn.embedding_lookup(W, X)    
    
    
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    print X[0]
    var = W.eval(session=sess)
    print var
main()
