import  pandas as pd
import numpy as np
import os
import tensorflow as tf
import  copy

def pre():

    df = pd.DataFrame.from_csv(os.getcwd() + "/../friends.csv")

    print(len(df))
    X = list(df['utterance'])
    Y = copy.deepcopy(X)
    X =X[:-1]
    Y = Y[1:]

    #print(len(X))
    #print(len(Y))
    return (X, Y)

def main():
    (X,Y) =pre()

main()
