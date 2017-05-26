"""
Contextual Sequence to Sequence model with attention
"""
from pydoc import locate

import tensorflow as tf

from seq2seq import decoders
from ghissubot.models.context_seq2seq import ContextSeq2Seq


class AttentionSeq2SeqWithContext(ContextSeq2Seq):
    """Sequence2Sequence model with attention mechanisms that also attends
    over a representation of context over prior utterances"""

    def __init__(self, params, mode, name="att_seq2seq"):
        super(AttentionSeq2SeqWithContext, self).__init__(params, mode, name)

    @staticmethod
    def default_params():
        params = ContextSeq2Seq.default_params().copy()
        params.update({
            "attention.class": "AttentionLayerBahdanau",
            "attention.params": {},  # Arbitrary attention layer parameters
            "bridge.class": "seq2seq.models.bridges.ZeroBridge",
            "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
            "encoder.params": {},  # Arbitrary parameters for the encoder
            "decoder.class": "seq2seq.decoders.AttentionDecoder",
            "decoder.params": {}  # Arbitrary parameters for the decoder
        })
        return params


    def _create_decoder(self, encoder_output, features, _labels):
        #Get attention function (Bahadnau or Luong)
        attention_class = locate(self.params["attention.class"]) or \
          getattr(decoders.attention, self.params["attention.class"])
        attention_layer = attention_class(
            params=self.params["attention.params"], mode=self.mode)

        # If the input sequence is reversed we also need to reverse
        # the attention scores.
        reverse_scores_lengths = None
        if self.params["source.reverse"]:
          reverse_scores_lengths = features["source_len"]
          if self.use_beam_search:
            reverse_scores_lengths = tf.tile(
                input=reverse_scores_lengths,
                multiples=[self.params["inference.beam_search.beam_width"]])

        #TODO: Make sure this is joining along the correct dimension
        #TODO: Might need to do some kind of projection here to make sizes align
        #Reshape context output
        context_batch_size = tf.shape(encoder_output.context_outputs)[0]
        context_embedding_width = tf.shape(encoder_output.context_outputs)[1]
        context_outputs_reshaped = tf.reshape(encoder_output.context_outputs, (context_batch_size, 1, context_embedding_width))

        #Append as extra hidden state to encoder hidden states
        attention_values = tf.concat([context_outputs_reshaped, encoder_output.attention_values], 1)
        attention_keys = tf.concat([context_outputs_reshaped, encoder_output.outputs], 1)
        attention_values_length = tf.add(encoder_output.attention_values_length, tf.constant(1))
        return self.decoder_class(
            params=self.params["decoder.params"],
            mode=self.mode,
            vocab_size=self.target_vocab_info.total_size,
            attention_values=attention_values,
            attention_values_length=attention_values_length,
            attention_keys=attention_keys,
            attention_fn=attention_layer,
            reverse_scores_lengths=reverse_scores_lengths)