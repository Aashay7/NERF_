import tensorflow as tf

# Implementation of Positional Encoding
def encoder_fn(p, L):
	gamma = [p]
	for i in range(L):
		gamma.append(tf.sin((2.0 ** i) * p))
		gamma.append(tf.cos((2.0 ** i) * p))
	
	gamma = tf.concat(gamma, axis=-1)
	return gamma