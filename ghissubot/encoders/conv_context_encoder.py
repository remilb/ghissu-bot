from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seq2seq.seq2seq.encoders.encoder import Encoder, EncoderOutput


class ConvContextEncoder(Encoder):
  """Encoder class that loads in an externally trained CNN classifer and encodes input
  sequences by passing them through it. Rips out the final hiddent state"""

  def __init__(self, params, mode, name="conv_context_encoder"):
      super(ConvContextEncoder, self).__init__(params, mode, name)
      if self.params["metagraph_filename"] == "":
          raise ValueError("Must provide metagraph file to load for {}!".format(self.__class__.__name__))
      if self.params["max_sequence_length"] == 0:
          raise ValueError("Must provide max sequence length for {}!".format(self.__class__.__name__))
      if self.params["input_name"] == "":
          raise ValueError("Must provide name of input tensor for {}!".format(self.__class__.__name__))
      if self.params["output_name"] == "":
          raise ValueError("Must provide name for output (hidden repr tensor) for {}!".format(self.__class__.__name__))
      if self.params["naming_prefix"] == "":
          raise ValueError("Must provide naming prefix used for loaded graph in class {}!".format(self.__class__.__name__))

  @staticmethod
  def default_params():
    return {
        "embedding_dim": 128,
        "metagraph_dir": "",
        "metagraph_filename": "",
        "max_sequence_length": 0,
        "vocab_size": 20816,
        "input_name": "",
        "output_name": "",
        "output_layer_size": 0,
        "naming_prefix": "",
        "freeze_graph": "True",
        "padding_token": "ENDPADDING"
    }

  def encode(self, inputs, sequence_length, **kwargs):
        #TODO: This is where we need to import metagraph and hook into it

        metagraph_file = self.params["metagraph_dir"] + self.params["metagraph_filename"]

        #TODO: Might need to make sure sequence lengths are all in order
        # Inspiration from this?
        # Slice source to max_len
        if self.params["max_sequence_length"] is not None:
            inputs = inputs[:, :self.params["max_sequence_length"]]
            sequence_length = tf.minimum(sequence_length, self.params["max_sequence_length"])



        #TODO: Bind input tensors (inputs) to the input tensor of loaded subgraph
        input_tensor_name = self.params["naming_prefix"] + '/' + self.params["input_name"] + ':0'
        input_map = {input_tensor_name: inputs}

        # Now import metagraph, remapping our inputs to the appropriate place
        tf.train.import_meta_graph(metagraph_file, input_map=input_map)
        current_graph = tf.get_default_graph()

        output_layer_name = self.params["naming_prefix"] + '/' + self.params["output_name"] + ':0'
        context_vector = current_graph.get_tensor_by_name(output_layer_name)

        #TODO: We need to make sure to freeze the output tensor so that gradients don't flow
        if self.params["freeze_graph"] == "True":
            context_vector = tf.stop_gradient(context_vector)

        # Note that we don't return an EncoderOutput like the other classes, just the context vector
        return context_vector
