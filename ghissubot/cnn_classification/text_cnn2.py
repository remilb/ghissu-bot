import tensorflow as tf
from seq2seq.configurable import Configurable
from seq2seq.data import vocab

class TextCNN(Configurable):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    @staticmethod
    def default_params():
        return {

            "embedding.size": 128,
            "embedding.init_scale": 0.04,
            "filter_sizes": list(map(int, "3,4,5,6,8".split(","))),
            "num_filters": 128,
            "dropout_keep_prob": 1.0,
            "l2_reg_lambda": 0.0,
            "checkpoint_dir": "",
            "checkpoint_filename": "",
            "allow_soft_placement": True,
            "log_device_placement": False,
            "cnn_source.max_seq_len": 25,
            "vocab_size": 26297,
            "layer_name": "context_layer:0",
            "num_classes": 43,
            "name_scope_of_convolutions": "conv-maxpool-",
            "vocab_source": "",
            "context_size": 512,
            "unique_prefix": ""
        }

    def __init__(self, params , mode, name = ""):
        with tf.variable_scope("inference_ops"):
            self.name = name
            super(TextCNN, self).__init__(params, mode)


            self.source_vocab_info = None
            if "vocab_source" in self.params and self.params["vocab_source"]:
                self.source_vocab_info = vocab.get_vocab_info(self.params["vocab_source"])
                print(self.source_vocab_info.path)

            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.string, [None, self.params["cnn_source.max_seq_len"]], name="cnn_input_utterances")
            self.input_y = tf.placeholder(tf.int32, [None, self.params["num_classes"]], name="cnn_input_labels")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            print(self.input_x)
            print(self.input_y)

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Vocab lookup ops
            with tf.variable_scope("vocab_lookup_ops"):
                # preprocess code
                source_vocab_to_id, source_id_to_vocab, source_word_to_count, _ = \
                  vocab.create_vocabulary_lookup_table(self.source_vocab_info.path)

                features = source_vocab_to_id.lookup(tf.identity(self.input_x))
                print("printing features shape", features.shape)


            # Add vocab tables to graph collection so that we can access them in
            # other places.
            #todo we probably need to remove this
            '''
            graph_utils.add_dict_to_collection({
                "cnn_source_vocab_to_id": self.source_vocab_to_id,
                "cnn_source_id_to_vocab": source_id_to_vocab,
                "cnn_source_word_to_count": source_word_to_count,
            }, "cnn_vocab_tables")
            #print(source_vocab_to_id.shape)'''

            '''end of preprocess code '''
            '''------------------------------------------------------------------'''

            '''source encoding definition'''
            self.source_embedding = tf.get_variable(
                name="cnn_embedding_layer",
                shape=[self.source_vocab_info.total_size, self.params["embedding.size"]],
                initializer=tf.random_uniform_initializer(
                    -self.params["embedding.init_scale"],
                    self.params["embedding.init_scale"]))
            '''end of source encoding'''
            '''------------------------------------------------------------------'''

            '''encode method'''
            self.embedded_chars = tf.nn.embedding_lookup(self.source_embedding,
                                                         features)
                #self.embedded_chars = tf.nn.embedding_lookup(self.W, input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            '''end of encode method'''
            '''------------------------------------------------------------------'''

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(self.params["filter_sizes"]):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                        #filter_shape = [filter_size, embedding_size, 1, num_filters]
                    filter_shape = [filter_size, self.params["embedding.size"], 1, self.params["num_filters"]]
                    W = tf.get_variable(name="W", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                    b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[self.params["num_filters"]]))
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
            print(self.params["num_filters"], len(self.params["filter_sizes"]))
            self.h_pool = tf.concat(pooled_outputs, 3)
            print(num_filters_total)
            print(self.h_pool.shape)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="flattened_layer")

            with tf.variable_scope("dropout_flatten"):
                self.h_pool_flat_dropout = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            self.feed_forward_layer = tf.contrib.layers.fully_connected(inputs=self.h_pool_flat_dropout,
                                                                        num_outputs=self.params["context_size"],
                                                                        scope="hidden_context_layer")
            # Add dropout

            with tf.variable_scope("dropout_fully_connected"):
                self.h_drop = tf.nn.dropout(self.feed_forward_layer, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.variable_scope("output"):
                W = tf.get_variable(
                    name="W",
                        # shape=[num_filters_total, self.params["num_classes"]],
                    shape=[params["context_size"], self.params["num_classes"]],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[self.params["num_classes"]]))
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.variable_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + self.params["l2_reg_lambda"] * l2_loss

            # Accuracy
            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            with tf.variable_scope("confusion_matrix"):
                # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.confusion_matrix = tf.confusion_matrix(tf.argmax(self.input_y, 1), self.predictions)

            variable_names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            for name in variable_names: print(name + "\n")