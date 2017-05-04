import numpy as np
import tensorflow as tf
import toy_examples.helpers

#x = [[5,7,8], [6,3], [3], [1]]

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

#Start session
sess = tf.Session()

#Placeholders for encoder inputs and decoder targets
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

#We'll also add this placeholder for now
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

#Let's deal with the embeddings for our inputs
#Here we randomly initialize our embeddings
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0),
                         dtype=tf.float32)

#Lookup embeddings for our inputs in parallel
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

####### Now Build the Encoder ########################
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

_, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True
) #Blank return is for encoder outputs, since we are doing seq2seq

###### Now do Decoder ##############################
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder"
)

# Now make the projection layer so that the outputs of the decoder time steps
# (of size equal to hidden units) can be projected back to vocab size so
# we can decode symbol
decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)
# Outputs projected from [max_time, batch_size, hidden_units] --> [max_time, batch_size, vocab_size]

####### Now the Optimizer ##################################
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())


###### Test Wiring ##############
batch_encoded, _ = helpers.batch([[6], [3, 4], [9, 8, 7]])
dec_ins, _ = helpers.batch(np.ones(shape=(3,1), dtype=np.int32),
                           max_sequence_length=4)

predictions = sess.run(decoder_prediction,
                       feed_dict={
                           encoder_inputs: batch_encoded,
                           decoder_inputs: dec_ins
                       })
print('decoder predictions: {}'.format(predictions))


#### Train Toy Task ###############
batch_size = 100
batches = helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)
print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)


def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

#### Train ######
loss_track = []
max_batches = 3001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')