"""
    28/11/2017 - Dott. Alessandro Maragno - model_compressor.py
    Brief: Open a pre-trained CNN model to apply SVD
	
    TO COMPLETE: Scrivere la parte relativa al retraining di ogni layer.
"""
from tensorflow.examples.tutorials.mnist import input_data

import net_utility as ut
import tensorflow as tf
import numpy as np
import os
import sys

tf.app.flags.DEFINE_string('orig_model', '', 'path where to find original model to load')
FLAGS = tf.app.flags.FLAGS

def conv2d(x, W) :
    return tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')

def maxpool_2x2(x) :
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=(1,2,2,1), padding='SAME')

def infer_model(var_names, model, input_feat) :
	"""
	Execute the forward pass of "model" on "input_feat"
	Input:
		var_names = list of names to select variables in "model"
		model = dictionary containing model variables initialized
		input_feat = placeholder for input feature maps to process
	Output:
		prediction = result of the forward pass application
	"""
	
	img_from_in_feat = tf.reshape(input_feat, [-1, 28, 28, 1])

	h_conv1 = tf.nn.relu(conv2d(img_from_in_feat, model[var_names[0]]) + model[var_names[1]])
	pool_1 = maxpool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(pool_1, model[var_names[2]]) + model[var_names[3]])
	pool_2 = maxpool_2x2(h_conv2)
	
	pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, model[var_names[4]]) + model[var_names[5]])

	prediction = tf.matmul(h_fc1, model[var_names[6]]) + model[var_names[7]]

	return prediction

def infer_approx_model(var_names, model, input_feat) :
	"""
	Execute the forward pass of "model" on "input_feat"
	Input:
		var_names = list of names to select variables in "model"
		model = dictionary containing model variables initialized
		input_feat = placeholder for input feature maps to process
	Output:
		prediction = result of the forward pass application
	"""
	
	img_from_in_feat = tf.reshape(input_feat, [-1, 28, 28, 1])

	vert_conv1 = conv2d(img_from_in_feat, model[var_names[0]])
	h_conv1 = tf.nn.relu(conv2d(vert_conv1, model[var_names[1]]) + model[var_names[2]])
	pool_1 = maxpool_2x2(h_conv1)

	vert_conv2 = conv2d(pool_1, model[var_names[3]])
	h_conv2 = tf.nn.relu(conv2d(vert_conv2, model[var_names[4]]) + model[var_names[5]])
	pool_2 = maxpool_2x2(h_conv2)
	
	pool_2_flat = tf.reshape(pool_2, [-1, model[var_names[6]].shape[0]])

	vert_fc1 = tf.matmul(pool_2_flat, model[var_names[6]])
	h_fc1 = tf.nn.relu(tf.matmul(vert_fc1, model[var_names[7]]) + model[var_names[8]])

	vert_fc2 = tf.matmul(h_fc1, model[var_names[9]])
	prediction = tf.matmul(vert_fc2, model[var_names[10]]) + model[var_names[11]]

	return prediction

def accuracy(labels, prediction) :
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(prediction, 1)), tf.float32))

def restore_compress_eval_save(model_dir) :
	"""
		Restore a pre-trained model (v. mnist_conv2fc2_no_estimator.py), then compress it (SVD) and evaluate
		accuracy of the approximated model so obtained. Compressed model is finally saved (have to insert the path).
		
		Input:
			model_dir = path with root name of the model to restore.
		Output:
			None
	"""

	model, var_names = ut.model_loader(model_dir, ret_vars=False)
	model_approx = {} # dictionary: key = name, val = V or H or bias
	model_approx_names = []

	for name in var_names:
		if len(model[name].shape) > 1 :
			print('compressing layer named {}'.format(name))
			V, H = ut.get_low_rank_approximation(model[name], name)
			model_approx_names.append(V.name)
			model_approx_names.append(H.name)
			model_approx[V.name] = V
			model_approx[H.name] = H
			print('{}={}, {}={}'.format(V.name, model_approx[V.name].shape, H.name, model_approx[H.name].shape))
			print('layer named {} compressed'.format(name))
		else :
			print('bias: {}'.format(name))
			model_approx_names.append(name)
			model_approx[name] = model[name] # bias

	approx_var_list = []
	for n in model_approx_names :
		approx_var_list.append(tf.Variable(model_approx[n]))

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	input_feat = tf.placeholder(tf.float32, [None, 784])
	labels = tf.placeholder(tf.float32, [None, 10])

	saver = tf.train.Saver(approx_var_list)

	main_session = tf.Session()

	print('Initializing all variables. . .')
	main_session.run(tf.global_variables_initializer())
	print('All variables initilized!')

	model.clear()
	var_names.clear()

	#accuracy_op = accuracy(labels, infer_model(var_names, model, input_feat))
	accuracy_op = accuracy(labels, infer_approx_model(model_approx_names, model_approx, input_feat))

	test_batch = mnist.test.next_batch(8000)

	accuracy_result = main_session.run(accuracy_op, 
									#feed_dict={input_feat: mnist.test.images, labels: mnist.test.labels})
									feed_dict={input_feat: test_batch[0], labels: test_batch[1]})

	#print('Test accuracy after reload: {}'.format(accuracy_result))
	print('Test accuracy after low-rank approximation: {}'.format(accuracy_result))

	saver.save(main_session, r'path\where\store\approx_model\model_root_name')

def main(_) :
	if len(sys.argv) != 3 or not sys.argv[-1][12:].startswith(('C:\\', '.')) \
	   or sys.argv[-2][12:].startswith(('C:\\', '.')) :
		print('Usage: restore_and_compress.py --orig_model=path\to\model\w_model_name')
		sys.exit(-1)
	
	orig_model_dir = sys.argv[-1][13:]

	if orig_model_dir.startswith('.') :
		orig_model_dir = os.path.abspath(orig_model_dir)

	restore_compress_eval_save(orig_model_dir)



if __name__ == '__main__' :
	tf.app.run()