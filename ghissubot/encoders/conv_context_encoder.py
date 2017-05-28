from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.data import vocab


class ConvContextEncoder(Encoder):
  """Encoder class that loads in an externally trained CNN classifer and encodes input
  sequences by passing them through it. Rips out the final hiddent state"""

  def __init__(self, params, mode, name="conv_context_encoder"):
      super(ConvContextEncoder, self).__init__(params, mode, name)
      if self.params["metagraph_filename"] == "":
          raise ValueError("Must provide metagraph file to load for {}!".format(self.__class__.__name__))
      if self.params["cnn_source.max_seq_len"] == 0:
          raise ValueError("Must provide max sequence length for {}!".format(self.__class__.__name__))
      if self.params["input_name"] == "":
          raise ValueError("Must provide name of input tensor for {}!".format(self.__class__.__name__))
      if self.params["output_name"] == "":
          raise ValueError("Must provide name for output (hidden repr tensor) for {}!".format(self.__class__.__name__))
      if self.params["naming_prefix"] == "":
          raise ValueError("Must provide naming prefix used for loaded graph in class {}!".format(self.__class__.__name__))
      self.special_source_vocab_info = vocab.get_vocab_info(self.params["vocab_path"])

  @staticmethod
  def default_params():
    return {
        "embedding.size": 128,
        "embedding.init_scale": 0.04,
        "filter_sizes": list(map(int, "3,4,5,6,8".split(","))),
        "num_filters": 128,
        "dropout_keep_prob": 1.0,
        "l2_reg_lambda": 0.0,
        "metagraph_dir": "",
        "metagraph_filename": "",
        "cnn_source.max_seq_len": 30,
        "vocab_size": 20816,
        "input_name": "",
        "output_name": "",
        "output_layer_size": 0,
        "naming_prefix": "",
        "freeze_graph": True,
        "padding_token": "ENDPADDING",
        "vocab_path": ""
    }

  def encode(self, inputs, sequence_length, **kwargs):
        #TODO: This is where we need to import metagraph and hook into it
        with tf.variable_scope("context_restore_prefix/our_special_prefix"):
            with tf.variable_scope("vocab"):
                '''preprocess code '''
                source_vocab_to_id, source_id_to_vocab, source_word_to_count, _ = \
                    vocab.create_vocabulary_lookup_table(self.special_source_vocab_info.path)

                features = source_vocab_to_id.lookup(tf.identity(inputs))

            '''source encoding definition'''
            source_embedding = tf.get_variable(
                name="cnn_W",
                shape=[self.special_source_vocab_info.total_size, self.params["embedding.size"]],
                initializer=tf.random_uniform_initializer(
                    -self.params["embedding.init_scale"],
                    self.params["embedding.init_scale"]))
            '''end of source encoding'''
            '''------------------------------------------------------------------'''

            '''encode method'''
            embedded_chars = tf.nn.embedding_lookup(source_embedding,
                                                         features)
            # self.embedded_chars = tf.nn.embedding_lookup(self.W, input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            '''end of encode method'''
            '''------------------------------------------------------------------'''

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(self.params["filter_sizes"]):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    # filter_shape = [filter_size, embedding_size, 1, num_filters]
                    filter_shape = [filter_size, self.params["embedding.size"], 1, self.params["num_filters"]]
                    W = tf.get_variable(name="W", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                    b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[self.params["num_filters"]]))
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.params["cnn_source.max_seq_len"] - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.params["num_filters"] * len(self.params["filter_sizes"])
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total], name="dummy")


            context_vector = tf.contrib.layers.fully_connected(inputs=h_pool_flat,
                                                                        num_outputs=512,
                                                                        scope="hidden_context_layer")

        #TODO: We need to make sure to freeze the output tensor so that gradients don't flow
        if self.params["freeze_graph"]:
            context_vector = tf.stop_gradient(context_vector)

        # logits_vector = current_graph.get_operation_by_name("context_restore_prefix/loss/SoftmaxCrossEntropyWithLogits")
        # tf.assign(logits_vector, tf.constant(0, shape=(64,10)))
        # Note that we don't return an EncoderOutput like the other classes, just the context vector
        return context_vector
