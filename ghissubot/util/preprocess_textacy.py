import math
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
    return tt.preprocess.preprocess_text(text=text, fix_unicode=True, lowercase=True, no_contractions=True, no_accents=True, transliterate=True)

def get_response_array(X):
    Y = copy.deepcopy(X)
    X = X[:-1]
    Y = Y[1:]
    return (X, Y)

def clean_punctuations(text):
    print (text)
    return re.sub("[^a-zA-Z .?!0-9,\']", "", text)
import  numpy as np
def preprocess(filename, utterance_train_filename, response_train_filename, utterance_dev_filename, response_dev_filename):
    df = pd.read_csv(filename, sep = ",")

    df = df.dropna(subset = ['utterance'])
    df['tokenised_sents'] = df['utterance'].apply(clean_punctuations).apply(textacy_preprocess).apply(tokenize)

    clean_utterances = np.array(df['tokenised_sents'])
    clean_utterances, response = get_response_array( clean_utterances)
    response = np.array(response)

    ''''#code to concat prev utterance
    delim = " "
    concat_utterance = []
    for i in range(len(clean_utterances)):
        concat_utterance.append(clean_utterances[i] + delim +  response[i])
    concat_utterance = concat_utterance[:-1]
    clean_utterances = pd.Series(data=concat_utterance)
    response_new = response[1:]
    response = pd.Series(data=response_new)
    '''



    division_boundary = math.floor(len(df)*0.9)
    clean_utterances = pd.Series(data=clean_utterances)
    response = pd.Series(data=response)
    clean_utterances_train = clean_utterances[:division_boundary]
    clean_utterances_dev = clean_utterances[division_boundary:]
    response_train = response[:division_boundary]
    response_dev = response[division_boundary:]

    clean_utterances_train.to_csv(utterance_train_filename, index=False)
    response_train.to_csv(response_train_filename, index=False)
    clean_utterances_dev.to_csv(utterance_dev_filename, index=False)
    response_dev.to_csv(response_dev_filename, index=False)
    clean_utterances.to_csv("utt.csv", index=False)

    print("done")



def preprocess_cnn(df, data_filename , label_filename, data_col, label_col):
    df = df.dropna(subset=[data_col])
    df['tokenised_sents'] = df[data_col].apply(clean_punctuations).apply(textacy_preprocess).apply(remove_quotes).apply(tokenize)

    clean_utterances = df['tokenised_sents']
    clean_utterances.to_csv(data_filename, index=False)

    df[label_col].to_csv(label_filename, index=False)
    print("done preprocessing")

def remove_quotes(text):
    return text.replace('"', '')


def main():
    # pass source and target filenames as commandline arguments
    filename = sys.argv[1]
    utterance_filename = sys.argv[2]
    response_filename = sys.argv[3]
    utterance_dev_filename = sys.argv[4]
    response_dev_filename = sys.argv[5]
    preprocess(filename, utterance_filename, response_filename, utterance_dev_filename, response_dev_filename)
    pass


def preprocess_text(text):
    return tokenize(remove_quotes(textacy_preprocess(clean_punctuations(text))))

main()
