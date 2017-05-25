from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seq2seq.seq2seq.encoders.encoder import Encoder, EncoderOutput


class ContextEncoder(Encoder):
  """A deep convolutional encoder, as described in
  https://arxiv.org/abs/1611.02344. The encoder supports optional positions
  embeddings."""


  def __init__(self, params, mode, name="conv_context_encoder"):
      super(ContextEncoder, self).__init__(params, mode, name)
      if self.params["metagraph_filename"] == "":
          raise ValueError("Must provide metagraph file to load for {}!".format(self.__class__.__name__))
      if self.params["sequence_length"] == 0:
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
        "output_cnn.units": 256,
        "embedding_dim": 128,
        "metagraph_dir": "",
        "metagraph_filename": "",
        "sequence_length": 0,
        "vocab_size": 20816,
        "input_name": "",
        "output_name": "",
        "naming_prefix": "",
        "freeze_graph": True
    }

  def encode(self, inputs, sequence_length, **kwargs):
        #TODO: This is where we need to import metagraph and hook into it

        metagraph_file = self.params["checkpoint_dir"] + self.params["metagraph_filename"]

        #TODO: Bind input tensors (inputs) to the input tensor of loaded subgraph
        # Might need to make sure sequence lengths are all in order
        input_tensor_name = self.params["naming_prefix"] + self.params["input_name"]
        input_map = {input_tensor_name: inputs}

        # Now import metagraph, remapping our inputs to the appropriate place
        tf.train.import_meta_graph(metagraph_file, input_map=input_map)
        current_graph = tf.get_default_graph()


        context_vector = current_graph.get_tensor_by_name(self.params["output_name"])

        #TODO: We need to make sure to freeze the subgraph so that gradients don't flow
        if self.params["freeze_graph"]:


        return EncoderOutput(outputs=context_vector)
