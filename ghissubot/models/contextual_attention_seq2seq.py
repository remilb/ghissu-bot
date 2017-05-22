import tensorflow as tf

from seq2seq.seq2seq.models import AttentionSeq2Seq


class ContextualAttentionSeq2Seq(AttentionSeq2Seq):
    """Sequence to Sequence model with two encoders, one for input sequence
       and one for preceding context. Attention mechanism attends to both encoders
       """

    def __init__(self, params, mode, name="context_att_seq2seq"):
        super(ContextualAttentionSeq2Seq, self).__init__(params, mode, name)

    def _create_decoder(self, encoder_output, features, _labels):
        """This needs to overide superclass method to construct decoder with attention that
        attends to context encoder as well"""
        pass