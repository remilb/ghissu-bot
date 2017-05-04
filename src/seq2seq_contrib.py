from util import preprocess as pp


class seq2seq_contrib:
    def __init__(self):
        #(vocab_size, embedding_dim, embedding , X, Y)
        ''' X is now an id array mapping to word embedding', Y is just the text of labels'''
        (self.vocab_size, self.embedding_dim, self.embedding , self.X, self.Y) = pp.read_from_csv()



