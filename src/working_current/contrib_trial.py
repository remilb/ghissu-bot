import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from model_new import Seq2SeqModel, train_on_copy_task
#import pandas as pd
import helpers

import warnings
warnings.filterwarnings("ignore")


tf.reset_default_graph()
tf.set_random_seed(1)

with tf.Session() as session:

    # with bidirectional encoder, decoder state size should be
    # 2x encoder state size
    model = Seq2SeqModel(encoder_cell=LSTMCell(10),
                         decoder_cell=LSTMCell(20),
                         vocab_size=400000,
                         embedding_size=50,
                         attention=True,
                         bidirectional=True,
                         custom_transform=True,
                         debug=False)

    session.run(tf.global_variables_initializer())

    train_on_copy_task(session, model,
                       length_from=3, length_to=8,
                       vocab_lower=2, vocab_upper=10,
                       batch_size=20,
                       max_batches=8000,
                       batches_in_epoch=1000,
                       verbose=True)
