from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pydoc import locate

import tensorflow as tf

from seq2seq.seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.seq2seq.encoders.pooling_encoder import _create_position_embedding


class ContextEncoder(Encoder):
  """A deep convolutional encoder, as described in
  https://arxiv.org/abs/1611.02344. The encoder supports optional positions
  embeddings.

  Params:
    attention_cnn.units: Number of units in `cnn_a`. Same in each layer.
    attention_cnn.kernel_size: Kernel size for `cnn_a`.
    attention_cnn.layers: Number of layers in `cnn_a`.
    embedding_dropout_keep_prob: Dropout keep probability
      applied to the embeddings.
    output_cnn.units: Number of units in `cnn_c`. Same in each layer.
    output_cnn.kernel_size: Kernel size for `cnn_c`.
    output_cnn.layers: Number of layers in `cnn_c`.
    position_embeddings.enable: If true, add position embeddings to the
      inputs before pooling.
    position_embeddings.combiner_fn: Function used to combine the
      position embeddings with the inputs. For example, `tensorflow.add`.
    position_embeddings.num_positions: Size of the position embedding matrix.
      This should be set to the maximum sequence length of the inputs.
  """

  # Placeholders for input, output and dropout


  def __init__(self, params, mode, name="conv_encoder"):

    super(ContextEncoder, self).__init__(params, mode, name)
    self.input_x = tf.placeholder(tf.int32, [None, self.params["sequence_length"]], name="input_x")
    self.input_y = tf.placeholder(tf.float32, [None, self.params["num_classes"]], name="input_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

  @staticmethod
  def default_params():
    return {
        "attention_cnn.units": 512,
        "attention_cnn.kernel_size": 3,
        "attention_cnn.layers": 15,
        "embedding_dropout_keep_prob": 0.8,
        "output_cnn.units": 256,
        "output_cnn.kernel_size": 3,
        "output_cnn.layers": 5,
        "position_embeddings.enable": True,
        "position_embeddings.combiner_fn": "tensorflow.multiply",
        "position_embeddings.num_positions": 100,
        "embedding_dim": 128,
        "filter_sizes": "3,4,5",
        "num_filters": 128,
        "dropout_keep_prob": 1.0,
        "l2_reg_lambda": 0.1,
        "checkpoint_dir": "data/switchboard/runs/1495511030/checkpoints/",
        "checkpoint_filename": "model-200",
        "allow_soft_placement": True,
        "log_device_placement": False,
        "sequence_length": 30,
        "vocab_size": 20816,
        "layer_name": "context_layer:0",
        "num_classes": 10,
        "name_scope_of_convolutions": "conv-maxpool-"
    }

  def encode(self, inputs, sequence_length):

        #Keeping track of l2 regularization loss (optional)

        context_graph_saver = tf.train.import_meta_graph("{}.meta".format(self.params["checkpoint_dir"] + self.params["checkpoint_file"]))
        context_graph = tf.get_default_graph()

        # Cutting of gradients for weights and biases on the convolution layers has to be performed and the model dumped

        context_layer = context_graph.get_tensor_by_name(self.params["layer_name"])

        return EncoderOutput(outputs = context_layer)