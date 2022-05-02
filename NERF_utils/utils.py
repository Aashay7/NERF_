from .config import (BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
import tensorflow as tf
import numpy as np

def get_focal_from_fov(fieldOfView, width):
	return 0.5 * width / tf.tan(0.5 * fieldOfView)


def render_image_depth(rgb, sigma, tVals):
	sigma = sigma[..., 0]
	
	delta = tVals[..., 1:] - tVals[..., :-1]
	deltaShape = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
	delta = tf.concat([delta, tf.broadcast_to([1e10], shape=deltaShape)], axis=-1)

	alpha = 1.0 - tf.exp(-sigma * delta)
	expTerm = 1.0 - alpha
	epsilon = 1e-10
	transmittance = tf.math.cumprod(expTerm + epsilon, axis=-1, exclusive=True)
	weights = alpha * transmittance
	
	image = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
	depth = tf.reduce_sum(weights * tVals, axis=-1)
	return (image, depth, weights)


def sample_pdf(tValsMid, weights, nF):
	weights += 1e-5
	pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
	cdf = tf.cumsum(pdf, axis=-1)
	cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)

	uShape = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, nF]
	u = tf.random.uniform(shape=uShape)

	indices = tf.searchsorted(cdf, u, side="right")
	below = tf.maximum(0, indices-1)
	above = tf.minimum(cdf.shape[-1]-1, indices)
	indicesG = tf.stack([below, above], axis=-1)

	cdfG = tf.gather(cdf, indicesG, axis=-1,batch_dims=len(indicesG.shape)-2)
	tValsMidG = tf.gather(tValsMid, indicesG, axis=-1, batch_dims=len(indicesG.shape)-2)

	denom = cdfG[..., 1] - cdfG[..., 0]
	denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
	t = (u - cdfG[..., 0]) / denom
	samples = (tValsMidG[..., 0] + t * (tValsMidG[..., 1] - tValsMidG[..., 0]))
	
	return samples


def get_translation_t(t):
	matrix = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, t],
		[0, 0, 0, 1],
	]
	matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)	
	return matrix

def get_rotation_phi(phi):
	matrix = [
		[1, 0, 0, 0],
		[0, tf.cos(phi), -tf.sin(phi), 0],
		[0, tf.sin(phi), tf.cos(phi), 0],
		[0, 0, 0, 1],
	]
	matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
	return matrix

def get_rotation_theta(theta):
	matrix = [
		[tf.cos(theta), 0, -tf.sin(theta), 0],
		[0, 1, 0, 0],
		[tf.sin(theta), 0, tf.cos(theta), 0],
		[0, 0, 0, 1],
	]
	matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
	return matrix

def pose_spherical(theta, phi, t):
	c2w = get_translation_t(t)
	c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
	c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
	c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
	return c2w