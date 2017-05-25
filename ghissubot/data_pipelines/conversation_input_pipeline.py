import tensorflow as tf

from seq2seq.seq2seq.data.input_pipeline import InputPipeline
from seq2seq.data import split_tokens_decoder, parallel_data_provider

from ghissubot.data_pipelines import split_utterances_decoder


class ConversationInputPipeline(InputPipeline):
    """Input pipeline for text files containing conversations, one line per turn"""
    #TODO: Right now only supports a really stupid file format for convenience of
    #implementation. Also it will only take in one context utterance
    @staticmethod
    def default_params():
        params = InputPipeline.default_params()
        params.update({
            "source_files": [],
            "target_files": [],
            "source_delimiter": " ",
            "source_utterance_delimiter": "|",
            "target_delimiter": " ",
        })
        return params

    def make_data_provider(self, **kwargs):
        #TODO: This needs to split context utterance from input
        decoder_source = split_utterances_decoder.SplitUtterancesDecoder(
            tokens_feature_name="source_tokens",
            length_feature_name="source_len",
            context_tokens_feature_name="context_tokens",
            context_length_feature_name="context_len",
            utterance_delimiter=self.params["source_utterance_delimiter"],
            append_token="SEQUENCE_END",
            delimiter=self.params["source_delimiter"])

        dataset_source = tf.contrib.slim.dataset.Dataset(
            data_sources=self.params["source_files"],
            reader=tf.TextLineReader,
            decoder=decoder_source,
            num_samples=None,
            items_to_descriptions={})

        dataset_target = None
        if len(self.params["target_files"]) > 0:
            decoder_target = split_tokens_decoder.SplitTokensDecoder(
                tokens_feature_name="target_tokens",
                length_feature_name="target_len",
                prepend_token="SEQUENCE_START",
                append_token="SEQUENCE_END",
                delimiter=self.params["target_delimiter"])

            dataset_target = tf.contrib.slim.dataset.Dataset(
                data_sources=self.params["target_files"],
                reader=tf.TextLineReader,
                decoder=decoder_target,
                num_samples=None,
                items_to_descriptions={})

        return parallel_data_provider.ParallelDataProvider(
            dataset1=dataset_source,
            dataset2=dataset_target,
            shuffle=self.params["shuffle"],
            num_epochs=self.params["num_epochs"],
            **kwargs)

    @property
    def feature_keys(self):
        return set(["context_tokens", "context_len", "source_tokens", "source_len"])

    @property
    def label_keys(self):
        return set(["target_tokens", "target_len"])