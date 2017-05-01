
import  pandas as pd
import numpy as np
import os
import tensorflow as tf
import  copy
def read_from_csv():
    df = pd.DataFrame.from_csv(os.getcwd() + "/../friends.csv")
    print(len(df))
    X = list(df['utterance'])
    Y = copy.deepcopy(X)
    X =X[:-1]
    Y = Y[1:]

    #print(len(X))
    #print(len(Y))
    return (X, Y)

def loadGloVe():
    filename = 'glove.6B.50d.txt'
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd



    pass

def main():
    sess = tf.Session()
    (X,Y) =read_from_csv()
    vocab,embd = loadGloVe()
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

main()
