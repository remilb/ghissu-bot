#! /usr/bin/env python

import csv
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from ghissubot.cnn_classification import data_helpers
from ghissubot.cnn_classification.text_cnn2 import TextCNN

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("checkpoint_dir", os.getcwd() + "/data/switchboard/runs/1495672830/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_string("checkpoint_filename", "model-200", "checkpoint filename to pick up from")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    print("Train mode")
    #x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    x_text, y_test = data_helpers.load_swbd_data(sequence_length=25)

else:
    print("Test mode")
    # Will not work since there is no padding being done
    x_text = ["a masterpiece four years in the making", "everything is off."]
    y_test = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Map data into vocabulary
# vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
# print(vocab_path)
checkpoint_file = os.path.join(FLAGS.checkpoint_dir, "", FLAGS.checkpoint_filename)
print(checkpoint_file)


print("\nEvaluating...\n")
print("\n\nFuck\n\n")

# # Evaluation
# # ==================================================
#
# #Trying to create new graph by itself
#
# params = {
# "embedding.size": 128,
# "embedding.init_scale": 0.04,
# "filter_sizes": list(map(int, FLAGS.filter_sizes.split(","))),
# "num_filters": 128,
# "dropout_keep_prob": 1.0,
# "l2_reg_lambda": 0.0,
# "checkpoint_dir": "",
# "checkpoint_filename": os.getcwd() + "/data/switchboard/runs/1495511030/checkpoints/",
# "allow_soft_placement": True,
# "log_device_placement": False,
# "cnn_source.max_seq_len": 25,
# "vocab_size": 25000,
# "layer_name": "context_layer:0",
# "num_classes": 10,
# "name_scope_of_convolutions": "conv-maxpool-",
# "vocab_source": os.getcwd() + "/data/switchboard/swbd_vocab",
# }
#
# # saver = tf.train.Saver()
#
# with tf.Graph().as_default():
#     session_conf = tf.ConfigProto(
#       allow_soft_placement=FLAGS.allow_soft_placement,
#       log_device_placement=FLAGS.log_device_placement,
#       device_count = {'GPU': 0}
#      )
#     sess = tf.Session(config=session_conf)
#     with sess.as_default():
#
#         cnn = TextCNN(params, "TRAIN", name="")
#         tf.train.Saver.restore(sess, FLAGS.checkpoint_dir + FLAGS.checkpoint_filename)
#         # Generate batches for one epoch
#         batches = data_helpers.batch_iter(x_text, FLAGS.batch_size, 1, shuffle=False)
#
#         softmax_tensor = sess.get_tensor_by_name('context_layer:0')
#         # # print(softmax_tensor)
#         # # Collect the predictions here
#         # all_predictions = []
#         # #softmax_tensor = sess.graph.get_tensor_by_name('context_layer:0')
#         #
#         #
#         for x_test_batch in batches:
#             softmax_tensor_value = sess.run(softmax_tensor,{cnn.input_x: x_test_batch, cnn.dropout_keep_prob: 1.0})
#
#             print(softmax_tensor_value, softmax_tensor_value.shape)
#         #     batch_predictions = []
#         #     all_predictions = np.concatenate([all_predictions, batch_predictions])
#
#         # =============================================================================================================
#
#         # Retrieve VGG inputs
#         # softmax_tensor = vgg_graph.get_tensor_by_name('context_layer:0')
#         # print(softmax_tensor)#.eval(session=tf.Session()))
#
#
# #
# # # =========================================================================
#
print("{}.meta".format(FLAGS.checkpoint_dir + FLAGS.checkpoint_filename))
vgg_saver = tf.train.import_meta_graph("{}.meta".format(FLAGS.checkpoint_dir + FLAGS.checkpoint_filename))
vgg_graph = tf.get_default_graph()


# ========================================================================
# vgg_saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
# # Access the graph
# vgg_graph = tf.get_default_graph()

if True:
# with vgg_graph:
    # session_conf = tf.ConfigProto(
    #   allow_soft_placement=FLAGS.allow_soft_placement,
    #   log_device_placement=FLAGS.log_device_placement,
    #   device_count = {'GPU': 1}
    #  )
    sess = tf.Session()#config=session_conf)


    #pool_3 = sess.graph.get_tensor_by_name('pool_3:0')
    #predictions, pool3_val = sess.run([softmax_tensor, pool_3],
    #                                  {'DecodeJpeg:0': image_data})


    # vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    # x_test = np.array(list(vocab_processor.transform(x_raw)))
    # print("{}.meta".format(checkpoint_file))
    # print("fuck\n\n\n")


    with sess.as_default():
        # Load the saved meta graph and restore variables


        # print("{}.meta".format(checkpoint_file))
        # #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # vgg_saver.restore(sess, checkpoint_file)
        #
        # # Get the placeholders from the graph by name
        input_x = vgg_graph.get_operation_by_name("input_x").outputs[0]
        input_y = vgg_graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = vgg_graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        #
        # # Tensors we want to evaluate
        # #predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        #
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(x_text, FLAGS.batch_size, 1, shuffle=False)

        softmax_tensor = vgg_graph.get_tensor_by_name('context_layer:0')
        # # print(softmax_tensor)
        # # Collect the predictions here
        # all_predictions = []
        # #softmax_tensor = sess.graph.get_tensor_by_name('context_layer:0')
        #
        #

        # # Generate batches
        # batches = data_helpers.batch_iter(
        #     list(zip(x_text, y_test)), FLAGS.batch_size, FLAGS.num_epochs)
        # # Training loop. For each batch...
        # for batch in batches:
        #     x_batch, y_batch = zip(*batch)


        for x_test_batch in batches:
            softmax_tensor = sess.run(softmax_tensor,{input_x: x_test_batch, dropout_keep_prob: 1.0})

            print(softmax_tensor, softmax_tensor.shape)
        #     batch_predictions = []
        #     all_predictions = np.concatenate([all_predictions, batch_predictions])

# =============================================================================================================

# Retrieve VGG inputs
#softmax_tensor = vgg_graph.get_tensor_by_name('context_layer:0')
#print(softmax_tensor)#.eval(session=tf.Session()))

# graph = tf.Graph()
#
# with graph.as_default():
#     session_conf = tf.ConfigProto(
#       allow_soft_placement=FLAGS.allow_soft_placement,
#       log_device_placement=FLAGS.log_device_placement,
#       device_count = {'GPU': 1}
#      )
#     sess = tf.Session(config=session_conf)
#
#
#     #pool_3 = sess.graph.get_tensor_by_name('pool_3:0')
#     #predictions, pool3_val = sess.run([softmax_tensor, pool_3],
#     #                                  {'DecodeJpeg:0': image_data})
#
#     print("{}.meta".format(checkpoint_file))
#     print("fuck\n\n\n")
#     with sess.as_default():
#         # Load the saved meta graph and restore variables
#
#
#         print("{}.meta".format(checkpoint_file))
#         saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#         saver.restore(sess, checkpoint_file)
#
#         # Get the placeholders from the graph by name
#         input_x = graph.get_operation_by_name("input_x").outputs[0]
#         # input_y = graph.get_operation_by_name("input_y").outputs[0]
#         dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
#
#         # Tensors we want to evaluate
#         predictions = graph.get_operation_by_name("output/predictions").outputs[0]
#
#         # Generate batches for one epoch
#         batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
#
#         # Collect the predictions here
#         all_predictions = []
#         softmax_tensor = sess.graph.get_tensor_by_name('context_layer:0')
#         for x_test_batch in batches:
#             softmax_tensor = sess.run(softmax_tensor,{input_x: x_test_batch, dropout_keep_prob: 1.0})
#
#             print(softmax_tensor, softmax_tensor.shape)
#             batch_predictions = []
#             all_predictions = np.concatenate([all_predictions, batch_predictions])


#
#
# # Print accuracy if y_test is defined
# if y_test is not None:
#     correct_predictions = float(sum(all_predictions == y_test))
#     print("Total number o
#
#
# f test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
#     #print("Confusion Matrix : {g}" .format)
#
# # Save the evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)

