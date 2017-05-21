import  nltk
import sys
import pandas as pd
import textacy as tt
import copy
import re


def tokenize(text):
    return " ".join(nltk.word_tokenize(text))
    pass

def textacy_preprocess(text):
    text = tt.preprocess.normalize_whitespace(text)
    return tt.preprocess.preprocess_text(text=text,fix_unicode=True, lowercase=True, no_contractions=True, no_accents=True, transliterate=True)

def get_response_array(X):
    Y = copy.deepcopy(X)
    X = X[:-1]
    Y = Y[1:]
    return (X, Y)

def clean_punctuations(text):
    print (text)
    return re.sub("[^a-zA-Z .?!0-9,\']", "", text)

def preprocess(filename, utterance_filename, response_filename):
    df = pd.read_csv(filename)

    df = df.dropna(subset = ['utterance'])
    df['tokenised_sents'] = df['utterance'].apply(clean_punctuations).apply(textacy_preprocess).apply(tokenize)

    clean_utterances = df['tokenised_sents']
    clean_utterances, response = get_response_array( clean_utterances)

    clean_utterances.to_csv(utterance_filename, index=False)
    response.to_csv(response_filename, index=False)

    print("done")



def preprocess_cnn(df, data_filename , label_filename, data_col, label_col):
    df = df.dropna(subset=[data_col])
    df['tokenised_sents'] = df[data_col].apply(clean_punctuations).apply(textacy_preprocess).apply(tokenize)

    clean_utterances = df['tokenised_sents']
    clean_utterances.to_csv(data_filename, index=False)

    df[label_col].to_csv(label_filename, index=False)
    print("done preprocessing")



def main():
    # pass source and target filenames as commandline arguments
    filename = sys.argv[1]
    utterance_filename = sys.argv[2]
    response_filename = sys.argv[3]
    preprocess(filename, utterance_filename, response_filename)
    pass

#main()
