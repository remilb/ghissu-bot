from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pydoc import locate

import tensorflow as tf

from seq2seq.encoders.encoder import Encoder
#from seq2seq.encoders.pooling_encoder import _create_position_embedding


class ConvEncoder(Encoder):
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

  def __init__(self, params, mode, name="conv_encoder"):
    super(ConvEncoder, self).__init__(params, mode, name)
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])

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
    }

  def encode(self, inputs, sequence_length):
    if self.params["position_embeddings.enable"]:
      positions_embed = _create_position_embedding(
          embedding_dim=inputs.get_shape().as_list()[-1],
          num_positions=self.params["position_embeddings.num_positions"],
          lengths=sequence_length,
          maxlen=tf.shape(inputs)[1])
      inputs = self._combiner_fn(inputs, positions_embed)

    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=inputs,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)

    with tf.variable_scope("cnn_a"):
      cnn_a_output = inputs
      for layer_idx in range(self.params["attention_cnn.layers"]):
        next_layer = tf.contrib.layers.conv2d(
            inputs=cnn_a_output,
            num_outputs=self.params["attention_cnn.units"],
            kernel_size=self.params["attention_cnn.kernel_size"],
            padding="SAME",
            activation_fn=None)
        # Add a residual connection, except for the first layer
        if layer_idx > 0:
          next_layer += cnn_a_output
        cnn_a_output = tf.tanh(next_layer)

    with tf.variable_scope("cnn_c"):
      cnn_c_output = inputs
      for layer_idx in range(self.params["output_cnn.layers"]):
        next_layer = tf.contrib.layers.conv2d(
            inputs=cnn_c_output,
            num_outputs=self.params["output_cnn.units"],
            kernel_size=self.params["output_cnn.kernel_size"],
            padding="SAME",
            activation_fn=None)
        # Add a residual connection, except for the first layer
        if layer_idx > 0:
          next_layer += cnn_c_output
        cnn_c_output = tf.tanh(next_layer)

    final_state = tf.reduce_mean(cnn_c_output, 1)

    return EncoderOutput(
        outputs=cnn_a_output,
        final_state=final_state,
        attention_values=cnn_c_output,
        attention_values_length=sequence_length)

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, params, mode, name="conv_encoder"):
        #super(ConvEncoder, self).__init__(params, mode, name)
        #self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])

    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")