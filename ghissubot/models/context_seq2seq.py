from pydoc import locate

import tensorflow as tf

from seq2seq.seq2seq.models import BasicSeq2Seq
from seq2seq import decoders
from seq2seq.graph_utils import templatemethod


class ContextSeq2Seq(BasicSeq2Seq):
    """Sequence to Sequence model with two encoders, one for input sequence
       and one for preceding context. Both encoder outputs are made available
       """

    def __init__(self, params, mode, name="context_att_seq2seq"):
        super(ContextSeq2Seq, self).__init__(params, mode, name)
        self.context_encoder_class = locate(self.params["context_encoder.class"])

    @staticmethod
    def default_params():
        params = BasicSeq2Seq.default_params().copy()
        params.update({
            "context_encoder.class": "", # Class to use for the context encoder
            "context_encoder.params": {}  # Arbitrary parameters for the context encoder
        })
        return params

    @templatemethod("encode_context")
    def encode_context(self, context_features, labels):
        context_encoder_fn = self.context_encoder_class(self.params["context_encoder.params"],
                                                        self.mode)
        #TODO: Figure out what else needs to be passed here
        return context_encoder_fn(context_features)

    def _create_decoder(self, encoder_output, features, _labels):
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

        return self.decoder_class(
            params=self.params["decoder.params"],
            mode=self.mode,
            vocab_size=self.target_vocab_info.total_size,
            attention_values=encoder_output.attention_values,
            attention_values_length=encoder_output.attention_values_length,
            attention_keys=encoder_output.outputs,
            attention_fn=attention_layer,
            reverse_scores_lengths=reverse_scores_lengths)

    #TODO: Need to override build so that we can split out the context features
    def _build(self, features, labels, params):
        # Pre-process features and labels
        features, labels = self._preprocess(features, labels)

        encoder_output = self.encode(features, labels)
        #TODO: Use right name for context features
        context_encoder_output = self.encode_context(features["context_feature"], labels)
        #TODO: Figure out the right way to provide context encoder output to decoder
        #TODO: Pack encoder_output and context_encoder_output into ExtendedEncoderOutput and pass to decode as usual
        packed_output = None
        decoder_output, _, = self.decode(packed_output, features, labels)

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            predictions = self._create_predictions(
                decoder_output=decoder_output, features=features, labels=labels)
            loss = None
            train_op = None
        else:
            losses, loss = self.compute_loss(decoder_output, features, labels)

            train_op = None
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                train_op = self._build_train_op(loss)

            predictions = self._create_predictions(
                decoder_output=decoder_output,
                features=features,
                labels=labels,
                losses=losses)

        # We add "useful" tensors to the graph collection so that we
        # can easly find them in our hooks/monitors.
        graph_utils.add_dict_to_collection(predictions, "predictions")

        return predictions, loss, train_op