"""Contains classes for various seq2seq models"""
import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from src.models.decorators import define_scope


class BasicSeq2Seq:
    """Basic seq2seq model - unidirection encode and decode"""
    def __init__(self):
        pass

    @define_scope
    def encoder(self):
        encoder_cell = tf_rnn.LSTM

    @define_scope
    def decoder(self):
        pass

    @define_scope
    def loss(self):
        pass