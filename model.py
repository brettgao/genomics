from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

LEARNING_RATE = 0.01
TRAINING_EPOCHS = 20
BATCH_SIZE = 256
DISPLAY_STEP = 1
examples_to_show = 10

N_INPUT = 784 # MNIST data input (img shape: 28*28)
N_HIDDEN = 100

FILENAME = 'data/out2.txt'

weights = {
  'encoder_h1': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN])),
  'decoder_h1': tf.Variable(tf.random_normal([N_HIDDEN, N_INPUT])),
}
biases = {
  'encoder_b1': tf.Variable(tf.random_normal([N_HIDDEN])),
  'decoder_b1': tf.Variable(tf.random_normal([N_INPUT])),
}

def load_data(filename):
  data = pd.read_csv('data/out2.txt', header=None)
  return data

def next_batch(batch_size):
  return mnist.train.next_batch(BATCH_SIZE)

def main():
  load_data(FILENAME)

  X = tf.placeholder("float", [None, N_INPUT])

  encoder_op = encoder(X)
  decoder_op = decoder(encoder_op)

  y_pred = decoder_op
  y_true = X

  cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
  optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(cost)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/BATCH_SIZE)
    for epoch in range(TRAINING_EPOCHS):
      for i in range(total_batch):
        batch_xs, batch_ys = next_batch()
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

      if epoch % DISPLAY_STEP == 0:
        print("Epoch:", '%04d' % (epoch+1), "{:.9f}".format(c))

      print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})

    # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # f.show()
    # plt.draw()
    # plt.waitforbuttonpress()

# Building the encoder
def encoder(x):
  # Encoder Hidden layer with sigmoid activation #1
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                 biases['encoder_b1']))
  return layer_1


# Building the decoder
def decoder(x):
  # Encoder Hidden layer with sigmoid activation #1
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                 biases['decoder_b1']))
  return layer_1


if __name__ == "__main__":
  main()

  