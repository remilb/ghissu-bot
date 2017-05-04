import copy

import numpy as np
import  pandas as pd
from tensorflow.contrib import learn

from util.embedding_models import load_glove


def read_from_csv():

    csv_fname = "/Users/shubhi/Public/CMPS296/friends.csv" #replace with local file loc
    glove_filename = '/Users/shubhi/Public/CMPS296/glove.6B/glove.6B.50d.txt'

    df = pd.DataFrame.from_csv(csv_fname)
    df = df.dropna(subset = ['utterance'])

    X = list(df['utterance'])
    X = clean_data(X, limit=20)

    Y = copy.deepcopy(X)
    X = X[:-1]
    Y = Y[1:]

    (vocab_size, embedding_dim, embedding , X) = embed_and_transform(X, glove_filename=glove_filename)
    return (vocab_size, embedding_dim, embedding , X, Y)


def embed_and_transform(X, glove_filename):
    max_sentence_length = 20
     #replace with local file loc
    vocab, embd = load_glove(glove_filename)

    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)

    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    X = np.array(list(vocab_processor.transform(X)))

    return (vocab_size, embedding_dim, embedding, X)


def shrink_vocab(X):
    from collections import Counter
    word_list = (" ".join(X)).split(" ")
    counter = Counter(word_list)
    #print (counter.count("the"))
    #return counter.most_common(10000)
    pass

def clean_data(X, limit):
    X = [" ".join(sentence.split("-")) for sentence in X]
    X = [" ".join(sentence.split(" ")[:limit]) for sentence in X]
    return X


read_from_csv()

'''
def tf_session(vocab_size, embedding_dim, embedding,  X):
    with tf.Session() as sess:

    # TF variable (placeholder)
        W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        embedding_init = W.assign(embedding_placeholder)

        req_embedded = tf.nn.embedding_lookup(W, X)

    # Call the session
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

        vectors = req_embedded.eval()
        print (vectors[0])
        pickle.dump(vectors, open("vectorized_input", "wb"))'''



