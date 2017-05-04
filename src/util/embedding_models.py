import nltk
import  numpy as np
import gensim.models.word2vec as wv

#glove_filename = "/Users/shubhi/Public/CMPS296/glove.6B/" + "glove.6B.100d.txt"
def load_glove(glove_filename):
    vocab = []
    embd = []
    file = open(glove_filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def word2vec(df):
    df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['utterance']), axis=1)
    import gensim, logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# train word2vec on the two sentences
    model = gensim.models.Word2Vec(df['tokenized_sents'], min_count=1)
    model.save("word2vec_model")
    return model


def gensim_word2vec(df):

    df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['utterance']), axis=1)
    model = wv.Word2Vec(df["tokenized_sents"], size=100, window=5, min_count=5, workers=4)
    model.save("word2vec")
    model = wv.Word2Vec.load("word2vec")
    #model.similarity("this", "is")
    model.init_sims(replace=True)
    return model
    #pass


def getEmbedding(sentence):
    gloveFile ="/Users/shubhi/Public/CMPS296/glove.6B/" + "glove.6B.100d.txt"
    model = loadGloveModel(gloveFile=gloveFile)
    #model = word2vec(df)
    #df['word2vec'] = np.array([])
    list = np.array([])

    for word in sentence:
        if word in model.values():
            list = np.append(list, model[word])

    return list
