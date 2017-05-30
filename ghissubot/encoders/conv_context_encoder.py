from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import gfile

from seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.data import vocab


class ConvContextEncoder(Encoder):
  """Encoder class that loads in an externally trained CNN classifer and encodes input
  sequences by passing them through it. Rips out the final hidden state"""

  def __init__(self, params, mode, name="conv_context_encoder"):
      super(ConvContextEncoder, self).__init__(params, mode, name)
      if self.params["frozen_graph_filename"] == "":
          raise ValueError("Must provide frozen graph protobuf file to load for {}!".format(self.__class__.__name__))
      if self.params["max_sequence_len"] == 0:
          raise ValueError("Must provide max sequence length for {}!".format(self.__class__.__name__))
      if self.params["input_tensor_name"] == "":
          raise ValueError("Must provide name of input tensor for {}!".format(self.__class__.__name__))
      if self.params["output_tensor_name"] == "":
          raise ValueError("Must provide name for output (hidden repr tensor) for {}!".format(self.__class__.__name__))
      if self.params["naming_prefix"] == "":
          raise ValueError("Must provide naming prefix used for loaded graph in class {}!".format(self.__class__.__name__))
      if self.params["table_init_op_name"] == "":
          raise ValueError("Must provide a name for table init op in subgraph for {}!".format(self.__class__.__name__))

  @staticmethod
  def default_params():
    return {
        "frozen_graph_dir": "",
        "frozen_graph_filename": "",
        "max_sequence_len": 0,
        "naming_prefix": "",
        "input_tensor_name": "",
        "output_tensor_name": "",
        "dropout_prob_name": "",
        "table_init_op_name": "",
        "output_layer_size": 0,
    }

  def encode(self, inputs, sequence_length, **kwargs):
        #TODO: This is where we need to import metagraph and hook into it

        frozen_graph_file = self.params["frozen_graph_dir"] + '/' + self.params["frozen_graph_filename"]

        #TODO: Might need to make sure sequence lengths are all in order
        if self.params["max_sequence_len"] is not None:
            inputs = inputs[:, :self.params["max_sequence_len"]]
            sequence_length = tf.minimum(sequence_length, self.params["max_sequence_len"])

        # Now we have to read in binary protobuf for frozen graph, then parse it to a GraphDef
        with gfile.FastGFile(frozen_graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Define our input map and output tensors of interest
        input_tensor_name = self.params["naming_prefix"] + '/' + self.params["input_tensor_name"] + ':0'
        input_map = {input_tensor_name: inputs}
        output_tensors = [self.params["naming_prefix"] + '/' + self.params["output_tensor_name"]]

        # Also want to return the table initializer op
        output_tensors.append(self.params["table_init_op_name"])

        # We also need to nullify the dropout layer
        dropout_name = self.params["naming_prefix"] + '/' + self.params["dropout_prob_name"]
        input_map[dropout_name] = tf.constant(1.0)

        # Now we import the above graph edf, remapping our inputs to the appropriate nodes
        # Note we index to zero to get the single output tensor we are requesting, as this returns a list
        # It also returns an Operation, so we grab its outputs property to get a tensor
        subgraph_nodes = tf.import_graph_def(graph_def, input_map=input_map, return_elements=output_tensors)
        context_vector = subgraph_nodes[0].outputs[0]
        table_initializer = subgraph_nodes[1]

        # Finally, we need to add the table initializers in the imported subgraph to the global table initializers collection
        # This ensures that the vocab lookup tables in the subgraph get initialized. This is so fucked up
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, table_initializer)

        # Now we need to get a handle to the outputs

        # Note that we don't return an EncoderOutput like the other classes, just the context vector
        return context_vector
