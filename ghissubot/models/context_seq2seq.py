from pydoc import locate

import tensorflow as tf

from seq2seq.seq2seq.models import BasicSeq2Seq
from seq2seq import decoders
from seq2seq.graph_utils import templatemethod


class ContextSeq2Seq(BasicSeq2Seq):
    """Basic Sequence to Sequence model with two encoders, one for input sequence
       and one for preceding context. Both encoder outputs are made available to decode
       related ops by packing into an ExtendedEncoderOutput. Nothing is done with the context
       encoder output in this class, thic class should function equivalentally to BasicSeq2Seq,
       besides exposing the context. New dictionary parameters added:
         Args:
    params: A dictionary of hyperparameters
      - context_encoder.class: An encoder class to use for the context encoder
      - context_encoder.params: Params to pass to context encoder class
       """

    def __init__(self, params, mode, name="context_seq2seq"):
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