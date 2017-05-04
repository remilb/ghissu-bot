import math

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import random
from util import helpers


class Seq2SeqModel():
    '''Seq2Seq model usign blocks from new `tf.contrib.seq2seq.'''

    PAD = 0
    EOS = 1
    bucket = []
    def __init__(self, encoder_cell, decoder_cell, vocab_size, embedding_size,
                 bidirectional=True,
                 attention=False,
                 debug=False):
        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell
        self._make_graph()

    def get_batch(X, vocab_size, batch_size):
        global PAD, EOS
        encoder_size, decoder_size  =  vocab_size
        encoder_inputs = []
        decoder_inputs = []
        for a in range(batch_size):
            encoder_inputs, decoder_inputs = random.choice(X)
            encoder_pad = [PAD] * (encoder_size - len(encoder_inputs))
            '''encoder inputs are padded and then reversed and cast into a list'''
            encoder_inputs.append(list(reversed(encoder_inputs +  encoder_pad)))

            '''decoder inputs get an extra GO/EOS symbol and are padded '''
            decoder_pad_size = decoder_size - len(decoder_inputs) -1  #because we have to add EOS
            decoder_inputs.append([EOS] +  decoder_inputs +  [PAD] * decoder_pad_size)
