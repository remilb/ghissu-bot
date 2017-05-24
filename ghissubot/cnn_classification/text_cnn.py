import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from seq2seq import graph_utils
from seq2seq.configurable import Configurable
from seq2seq.data import vocab
from seq2seq.graph_utils import templatemethod


class TextCNN(Configurable):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, params , mode , name = ""):
        '''cnn_source.max_seq_len, num_classes, vocab_size,
        embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):'''
        self.name = name
        super(TextCNN, self).__init__(params, mode)

        self.source_vocab_info = None
        if "vocab_source" in self.params and self.params["vocab_source"]:
            print("here")
            self.source_vocab_info = vocab.get_vocab_info(self.params["vocab_source"])
            print(self.source_vocab_info.path)

        self.input_x = tf.placeholder(tf.string, [None, self.params["cnn_source.max_seq_len"]], name="input_x_string")
        #self.input_x = tf.placeholder(tf.string, [None], name="input_x_string")
        self.input_y = tf.placeholder(tf.int32, [None, self.params["num_classes"]], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        (features, labels) = self._preprocess(features=self.input_x, labels=self.input_y)
        self.encode(features=features)
        self.init_model()

    @staticmethod
    def default_params():
        return {

            "embedding.size": 128,
            "embedding.init_scale": 0.04,
            "filter_sizes": "3,4,5",
            "num_filters": 128,
            "dropout_keep_prob": 1.0,
            "l2_reg_lambda": 0.0,
            "checkpoint_dir": "",
            "checkpoint_filename": "",
            "allow_soft_placement": True,
            "log_device_placement": False,
            "cnn_source.max_seq_len": 30,
            "vocab_size": 26297,
            "layer_name": "context_layer:0",
            "num_classes": 43,
            "name_scope_of_convolutions": "conv-maxpool-",
            "vocab_source": "",
        }


    @property
    @templatemethod("source_embedding")
    def source_embedding(self):
        """Returns the embedding used for the source sequence.
        """
        return tf.get_variable(
            name="cnn_W",
            shape=[self.source_vocab_info.total_size, self.params["embedding.size"]],
            initializer=tf.random_uniform_initializer(
                -self.params["embedding.init_scale"],
                self.params["embedding.init_scale"]))


    @templatemethod("encode")
    def encode(self, features):
        self.embedded_chars = tf.nn.embedding_lookup(self.source_embedding,
                                                 features)
        #self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #todo replace this
        '''encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
        return encoder_fn(source_embedded, features["source_len"])'''

    def init_model(self):
        # Placeholders for input, output and dropout
        # todo replace placeholders by tensors


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)



        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate([3,4,5]): #self.params["filter_sizes"]):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                #filter_shape = [filter_size, embedding_size, 1, num_filters]
                filter_shape = [filter_size, self.params["embedding.size"], 1, self.params["num_filters"]]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.params["num_filters"]]), name="b")
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
                    ksize=[1, self.params["cnn_source.max_seq_len"] - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.params["num_filters"] * len(self.params["filter_sizes"])
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="context_layer")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.params["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.params["num_classes"]]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.params["l2_reg_lambda"] * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("confusion_matrix"):
           # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.confusion_matrix = tf.confusion_matrix(tf.argmax(self.input_y, 1), self.predictions)


    def _preprocess(self, features, labels):
        """Model-specific preprocessing for features and labels:
    
        - Creates vocabulary lookup tables for source and target vocab
        - Converts tokens into vocabulary ids
        """
        # Create vocabulary lookup for source
        source_vocab_to_id, source_id_to_vocab, source_word_to_count, _ = \
          vocab.create_vocabulary_lookup_table(self.source_vocab_info.path)

        # Add vocab tables to graph collection so that we can access them in
        # other places.
        #todo we probably need to remove this
        graph_utils.add_dict_to_collection({
            "cnn_source_vocab_to_id": source_vocab_to_id,
            "cnn_source_id_to_vocab": source_id_to_vocab,
            "cnn_source_word_to_count": source_word_to_count,
        }, "cnn_vocab_tables")

        # Slice source to max_len
        #if self.params["cnn_source.max_seq_len"] is not None:
          #features["cnn_source_tokens"] = features["cnn_source_tokens"][:, :self.params[ "cnn_source.max_seq_len"]]
          # slicing the data - max sequence length
          #features = features[:, :self.params["cnn_source.max_seq_len"]]
          #junk = 0
          #features["cnn_source_len"] = tf.minimum(features["cnn_source_len"],self.params["cnn_source.max_seq_len"])
          #todo change this to assignment / initialisation
          #features["cnn_source_len"] = tf.minimum( self.params["cnn_source.max_seq_len"], self.params["cnn_source.max_seq_len"])

        # Look up the source ids in the vocabulary
        features = source_vocab_to_id.lookup(features)
        print("printing features shape" , features.shape)
        #Might not need
        #features["cnn_source_len"] = tf.to_int32(features["cnn_source_len"])
        # tf.summary.histogram("cnn_source_len", tf.to_float(features["cnn_source_len"]))
        # if labels is None:
        #   return features, None

        # labels = labels.copy()
        # added by us : converting labels to one hot
        #labels = tf.one_hot(labels, self.params["num_classes"])


        # Keep track of the number of processed tokens, probably dont need here

        # num_tokens = tf.reduce_sum(features["cnn_source_len"])
        # token_counter_var = tf.Variable(0, "cnn_tokens_counter")
        # total_tokens = tf.assign_add(token_counter_var, num_tokens)
        # tf.summary.scalar("num_tokens", total_tokens)
        #
        # with tf.control_dependencies([total_tokens]):
        #   features["cnn_source_tokens"] = tf.identity(features["cnn_source_tokens"])

        # # Add to graph collection for later use
        # graph_utils.add_dict_to_collection(features, "cnn_features")
        # if labels:
        #   graph_utils.add_dict_to_collection(labels, "cnn_labels")

        return features, labels

    # def build(self, features, labels, params):
    #     features, labels = self._preprocess(features, labels)
    #     if self.mode == tf.contrib.learn.ModeKeys.INFER:
    #         predictions = self._create_predictions(
    #              features=features, labels=labels)
    #         loss = None
    #         train_op = None
    #     else:
    #         losses, loss = self.compute_loss(features, labels)
    #
    #         train_op = None
    #         if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
    #             train_op = self._build_train_op(loss)
    #
    #         predictions = self._create_predictions(
    #             features=features,
    #             labels=labels,
    #             losses=losses)

        # We add "useful" tensors to the graph collection so that we
        # can easly find them in our hooks/monitors.
        graph_utils.add_dict_to_collection(predictions, "predictions")

        return predictions, loss, train_op

        pass

