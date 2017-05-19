from util import preprocess as pp
import tensorflow as tf
from model import Seq2SeqModel
import tensorflow.contrib.seq2seq as tf_s2s
from tensorflow.contrib.rnn import LSTMCell, GRUCell

class seq2seq_contrib:
    def __init__(self):
        #(vocab_size, embedding_dim, embedding , X, Y)
        ''' X is now an id array mapping to word embedding', Y is just the text of labels'''
        (self.vocab_size, self.embedding_dim, self.embedding , self.X, self.Y) = pp.read_from_csv()
        self.hidden_units_encoder = 10 #number of hidden units


    def tf_session(self):
        with tf.session() as session:
            model =  Seq2SeqModel(
                        encoder_cell=LSTMCell(input_size=self.hidden_units_encoder),
                        decoder_cell=LSTMCell(input_size=self.hidden_units_encoder),
                        vocab_size=self.vocab_size,
                        attention=True,
                        bidirectional=True,
                        debug=False
            )

         # Word Embedding Initialiser
            W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_dim]),
                        trainable=False, name="W")
            embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_dim])
            embedding_init = W.assign(embedding_placeholder)
            req_embedded = tf.nn.embedding_lookup(W, self.X)

            session.run(tf.global_variables_initializer())




            #global vars initialised

            # Call the session
            #session.run(embedding_init, feed_dict={embedding_placeholder: self.embedding})
            #vectors = req_embedded.eval()




