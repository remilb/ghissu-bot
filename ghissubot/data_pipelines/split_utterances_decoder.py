import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_decoder

from seq2seq.seq2seq.data.split_tokens_decoder import SplitTokensDecoder

class SplitUtterancesDecoder(SplitTokensDecoder):
    def __init__(self,
                 utterance_delimiter="|",
                 context_tokens_feature_name="context_tokens",
                 context_length_feature_name="context_length",
                 **kwargs):
        super(SplitUtterancesDecoder, self).__init__(kwargs)
        self.utterance_delimiter = utterance_delimiter
        self.context_tokens_feature_name = context_tokens_feature_name
        self.context_length_feature_name=context_length_feature_name

    def decode(self, data, items):
        decoded_items = {}

        #split context utterance from input utterance
        context_and_utterance = tf.string_split([data], delimiter=self.utterance_delimiter).values

        utterance_feature_names = super(SplitUtterancesDecoder, self).list_items()
        utterance_features = super(SplitUtterancesDecoder, self).decode(context_and_utterance[1],
                                                                        utterance_feature_names)
        context_tokens = tf.string_split([context_and_utterance[0]], delimiter=self.delimiter).values
        context_tokens_length = tf.size(context_tokens)

        return [utterance_features[0], utterance_features[1], context_tokens, context_tokens_length]

    def list_items(self):
        return [self.tokens_feature_name, self.length_feature_name,
                self.context_tokens_feature_name, self.context_length_feature_name]
