import copy
import nltk
import numpy as np
import  pandas as pd
from tensorflow.contrib import learn
import os
from embedding_models import load_glove


def read_from_csv():

    csv_fname = "/Users/shubhi/Public/CMPS296/friends.csv" #replace with local file loc
    #csv_fname="/Users/shubhi/Public/CMPS296/friends_sample.csv"
    glove_filename = '/Users/shubhi/Public/CMPS296/glove.6B/glove.6B.50d.txt'

    df = pd.DataFrame.from_csv(csv_fname)
    df = df.dropna(subset = ['utterance'])

    X = list(df['utterance'])
    #X = X[:100]
    X = clean_data(X, limit=20)

    Y = copy.deepcopy(X)
    X = X[:-1]
    Y = Y[1:]

    (vocab_size, embedding_dim, embedding , X, Y_input) = embed_and_transform(X, Y, glove_filename=glove_filename)

    return (vocab_size, embedding_dim, embedding , X, Y_input, Y_input)

def read_from_csv_with_custom_transform():

    csv_fname = "/Users/shubhi/Public/CMPS296/friends_sample.csv" #replace with local file loc
    #csv_fname = "/Users/shubhi/Public/CMPS296/friends.csv"
    csv_fname=os.getcwd()+ "/../cornell_data.csv"
    glove_filename = '/Users/shubhi/Public/CMPS296/glove.6B/glove.6B.50d.txt'

    df = pd.DataFrame.from_csv(csv_fname)
    df = df.dropna(subset = ['utterance'])

    X = list(df['utterance'])
    X = clean_data(X, limit=20)
    #X = X[:100]
    Y = copy.deepcopy(X)
    word_to_id_mapping = custom_transorm(X)
    word_to_id_mapping['PAD'] = 0
    word_to_id_mapping['EOS'] = 1
    inv_map = {v: k for k, v in word_to_id_mapping.items()}

    X = X[:-1]
    Y = Y[1:]
    Y_transform =  map_to_indices(Y, word_to_id_mapping)

    return (map_to_indices(X , word_to_id_mapping), Y_transform, Y_transform, inv_map, len(inv_map)+1 )


def embed_and_transform(X, Y,  glove_filename):
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
    Y = np.array(list(vocab_processor.transform(Y)))
    return (vocab_size, embedding_dim, embedding, X, Y)

def custom_transorm(X):
    word_to_id_mapping = {}
    map_id =2
    for sentence in X:
        sentence_map=[]
        for word in sentence:
            if word not in word_to_id_mapping.keys():
                word_to_id_mapping[word] = map_id
                map_id += 1
            sentence_map.append(word_to_id_mapping[word])
    return word_to_id_mapping

def map_to_indices(X, map):
     X_indices = []
     for sentence in X:
        sentence_map=[]
        for word in sentence:
             sentence_map.append(map[word])
        X_indices.append(np.asanyarray(sentence_map))
     return np.asanyarray(X_indices)


def shrink_vocab(X):
    from collections import Counter
    word_list = (" ".join(X)).split(" ")
    counter = Counter(word_list)
    #print (counter.count("the"))
    #return counter.most_common(10000)
    pass

def clean_data(X, limit):
    X = [" ".join(sentence.split("-")) for sentence in X]
    X_new = []

    for sentence in X:
        if(len(sentence.split(" ")) > limit):
            sentence = " ".join(sentence.split(" ")[0:limit])
        X_new.append(sentence)

    print("X in preprocess")

    X = [ nltk.word_tokenize(x) for x in X_new]
    print ("done with tokenising")

    #print (X)
    return X


#read_from_csv()




