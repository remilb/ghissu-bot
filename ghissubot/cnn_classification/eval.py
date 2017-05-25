#! /usr/bin/env python

import os

import tensorflow as tf

from ghissubot.cnn_classification import data_helpers

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("checkpoint_dir", os.getcwd() + "/data/switchboard/runs/1495709993/checkpoints/",
                       "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_string("checkpoint_filename", "model-100", "checkpoint filename to pick up from")
tf.flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs (default: 200)")
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
    # x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
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

print("{}.meta".format(FLAGS.checkpoint_dir + FLAGS.checkpoint_filename))
vgg_saver = tf.train.import_meta_graph("{}.meta".format(FLAGS.checkpoint_dir + FLAGS.checkpoint_filename))
vgg_graph = tf.get_default_graph()

with vgg_graph.as_default():
    sess = tf.Session()  # config=session_conf)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        tf.tables_initializer().run()
        vgg_saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        # with tf.name_scope("context_restore_prefix"):

        input_x = vgg_graph.get_operation_by_name("context_restore_prefix/cnn_input_placeholder").outputs[0]
        input_y = vgg_graph.get_operation_by_name("context_restore_prefix/input_y").outputs[0]
        dropout_keep_prob = vgg_graph.get_operation_by_name("context_restore_prefix/dropout_keep_prob").outputs[0]
        #hook_x = vgg_graph.get_tensor_by_name('cnn_input_hook:0')
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(x_text, FLAGS.batch_size, 1, shuffle=False)

        softmax_tensor = vgg_graph.get_tensor_by_name('context_restore_prefix/hidden_context_layer:0')
        # input_x = vgg_graph.get_tensor_by_name('input_x_string:0')
        # input_y = vgg_graph.get_tensor_by_name('input_y:0')
        # dropout_keep_prob = vgg_graph.get_tensor_by_name('dropout_keep_prob:0')

        for x_test_batch in batches:
            softmax_array = sess.run(softmax_tensor, feed_dict={input_x: x_test_batch, dropout_keep_prob: 1.0})

            print(softmax_array.shape)
