"""
	28/11/2017 - Dott. Alessandro Maragno - mnist_conv2fc2_no_estimator.py
	Building MNIST classifier without using tf.estimator.Estimator() and saving it after training
"""
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import os
import sys

tf.app.flags.DEFINE_integer('training_steps', 20000, '# of training iterations')
tf.app.flags.DEFINE_integer('version', 1, 'model version')
tf.app.flags.DEFINE_string('model_dir',
						   'default\path\where\to\save\model\model_root_name',
						   'location of the model')

FLAGS = tf.app.flags.FLAGS

def conv2d(x, W) :
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool_2x2(x, name) :
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

def main(_) :
	# ARGUMENT PARSING
	if len(sys.argv) < 2 or not sys.argv[-1].startswith('--model_dir=C:/') :
		print('Usage: mnist_conv2fc2_no_estimator.py [--training_steps=x] [--version=y] --model_dir=C:/path')
		sys.exit(-1)
	if FLAGS.training_steps <= 0 :
		print('Error: insert a positive # of training iterations')
		sys.exit(-1)
	if FLAGS.version <= 0 :
		print('Error: insert a positive version number for the model')
		sys.exit(-1)

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
	bias_initializer = tf.constant_initializer(0.1)

	# placeholders
	features = tf.placeholder(tf.float32, [None, 784])
	dropout_p = tf.placeholder(tf.float32)
	labels = tf.placeholder(tf.float32, [None, 10])

	# 1st conv layer
	W_conv1 = tf.get_variable('W_conv1', [5, 5, 1, 32], initializer=weight_initializer)
	b_conv1 = tf.get_variable('b_conv1', [32], initializer=bias_initializer)

	image = tf.reshape(features, [-1, 28, 28, 1]) # reshape x which is a vector with 28x28 greylevel pixels

	h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1, name='h_conv1') # apply non-linearity (activate (fire) neuron)
	h_pool1 = maxpool_2x2(h_conv1, 'h_pool1') # 2x2 subsampling

	# 2nd conv layer
	W_conv2 = tf.get_variable('W_conv2', [5, 5, 32, 64], initializer=weight_initializer)
	b_conv2 = tf.get_variable('b_conv2', [64], initializer=bias_initializer)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')
	h_pool2 = maxpool_2x2(h_conv2, 'h_pool2')

	# 1st fc layer
	W_fc1 = tf.get_variable('W_fc1', [7*7*64, 1024], initializer=weight_initializer)
	b_fc1 = tf.get_variable('b_fc1', [1024], initializer=bias_initializer)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')

	# Dropout layer
	h_fc1_drop = tf.nn.dropout(h_fc1, dropout_p, name='h_fc1_drop')

	# 2nd fc layer
	W_fc2 = tf.get_variable('W_fc2', [1024, 10], initializer=weight_initializer)
	b_fc2 = tf.get_variable('b_fc2', [10], initializer=bias_initializer)

	# READOUT PREDICTION
	prediction = tf.matmul(h_fc1_drop, W_fc2, name='prediction') + b_fc2

	# LOSS FUNCTION TO MINIMIZE
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=prediction), name='cross_entropy_loss')

	# OPTIMIZER
	train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

	# ACCURACY
	correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(prediction, 1), name='correct_prediction')
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver(save_relative_paths=True)

	for step in range(FLAGS.training_steps) :
		batch = mnist.train.next_batch(50)

		if step % 100 == 0 :
			accuracy_val = accuracy.eval(feed_dict={features: batch[0], labels: batch[1], dropout_p: 1.0}, session=sess)
			print('training accuracy at step {}: {}'.format(step, accuracy_val))

		train_op.run(feed_dict={features: batch[0], labels: batch[1], dropout_p: 0.5}, session=sess)

	print('test accuracy: {}'.format(accuracy.eval(feed_dict={features: mnist.test.images,
															   labels: mnist.test.labels,
															   dropout_p: 1.0},
												   session=sess)))

	print('Saving trained model. . .')
	
	export_dir = sys.argv[-1][12:] + '_v' + str(FLAGS.version)
	saver.save(sess, export_dir, global_step=FLAGS.training_steps)

if __name__ == '__main__':
	tf.app.run()