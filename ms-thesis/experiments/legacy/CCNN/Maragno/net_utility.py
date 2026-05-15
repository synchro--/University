"""
	20cnet_utility.py - Dott. Alessandro Maragno - 08/12/2017

	Module including some custom utility function to import
"""

import tensorflow as tf
import numpy as np

def model_loader(model_dir, ret_vars=True) :
	"""
	Loads the model from "model_dir"
	Input:
		model_dir = path to the model files, including the root name of the model at the end.
		ret_vars = boolean. If True, return Tensorflow Variable, else return Numpy tensor. The latter is
					useful if there's work to do on tensor data (e.g. low-rank approximation)
	Output:
		init_trainable = dictionary containing all trainable variables initialized with the loaded data
		names = list of names to use as keys of init_trainable
	"""

	print('Loading model from: {} . . .'.format(model_dir))

	tf.train.import_meta_graph(model_dir+'.meta')
	trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

	ckpt_reader = tf.train.NewCheckpointReader(model_dir)

	init_trainable = {}
	names = [v.name[:-2] for v in trainable]

	for n in names :
		if ret_vars :
			curr_tensor = ckpt_reader.get_tensor(n)
			init_trainable[n] = tf.get_variable(n,
									   curr_tensor.shape,
									   dtype=tf.float32,
									   initializer=tf.constant_initializer(curr_tensor))
		else :
			init_trainable[n] = ckpt_reader.get_tensor(n)

	print('Model successfully loaded!')

	return init_trainable, names

def tensor_to_mat(tensor) :
	"""
	Flatten a tensor into a matrix by spreading input channels kernels on rows and
	kernels composing the filter bank on columns.

	Input:
		tensor = tensor to flat
	Output:
		mat = resulting matrix
	"""
	tensor = np.array(tensor)

	in_fmaps = int(tensor.shape[2])
	k_size = int(tensor.shape[0])
	out_fmaps = int(tensor.shape[3])

	mat = np.zeros([in_fmaps*k_size, out_fmaps*k_size])

	for i in range(in_fmaps) :
		for r in range(k_size) :
			for c in range(k_size) :
				for o in range(out_fmaps) :
					mat[(i*k_size + r), (o*k_size + c)] = tensor[r, c, i, o]

	return mat

def get_low_rank_approximation(tensor, name) :
	"""
	Get low-rank approximation of a tensor defined in numpy
	Input:
		tensor = numpy array containing weights
	Output:
		[V, H] = low-rank approximation from "tensor". Horizontal and vertical filter. Returned as a list.
	
	H.shape = [1, k_cols, rank, output_fmaps]
	V.shape = [k_rows, 1, input_fmaps, rank]
	k_rows=k_cols=k_size
	rank < (k_size*out_fmaps*in_fmaps)/(out_fmaps+in_fmaps)
	"""
	tensor = np.array(tensor)

	if len(tensor.shape) == 4:
		k_size = int(tensor.shape[0])
		in_fmaps = int(tensor.shape[2])
		out_fmaps = int(tensor.shape[3])
		print('conv layer shape: {}'.format(tensor.shape))
		rank = int(k_size*in_fmaps*out_fmaps/(out_fmaps+in_fmaps))

		flatten_tensor = tensor_to_mat(tensor)

		H = np.zeros([1, k_size, rank, out_fmaps])
		V = np.zeros([k_size, 1, in_fmaps, rank])

		u, s, vt = np.linalg.svd(flatten_tensor)

		s_approx = np.diag(np.sqrt(s[ : rank]))
		u_approx = u[:, : rank]
		vt_approx = vt[: rank, :]

		for c in range(in_fmaps) :
			for k in range(rank) :
				for r in range(k_size) :
					V[r, 0, c, k] = u_approx[c*k_size + r, k] * s_approx[k, k]

		v_approx = np.transpose(vt_approx)

		for o in range(out_fmaps) :
			for k in range(rank) :
				for r in range(k_size) :
					H[0, r, k, o] = v_approx[o*k_size + r, k] * s_approx[k, k]
		
		print('separated conv layer shapes: v = {}, h = {}'.format(V.shape, H.shape))

	else : # fully connected layer is a matrix
		print('fully connected layer shape: {}'.format(tensor.shape))
		k_rows = int(tensor.shape[0])
		k_cols = int(tensor.shape[1])
		in_fmaps = 1
		out_fmaps = 1
		
		rank = int((k_cols*in_fmaps*out_fmaps)/(out_fmaps+in_fmaps))

		u, s, vt = np.linalg.svd(tensor)

		s_approx = np.diag(np.sqrt(s[ : rank]))
		u_approx = u[:, : rank]
		vt_approx = vt[: rank, :]

		H = np.matmul(s_approx, vt_approx)
		V = np.matmul(u_approx, s_approx)

		print('fully connected separated layer shapes: v = {}, h = {}'.format(V.shape, H.shape))

	return [tf.get_variable(name+'_v', V.shape, dtype=tf.float32, initializer=tf.constant_initializer(V)),
		 tf.get_variable(name+'_h', H.shape, dtype=tf.float32, initializer=tf.constant_initializer(H))]

