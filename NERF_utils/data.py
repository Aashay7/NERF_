from tensorflow.io import read_file
from tensorflow.image import (decode_jpeg, convert_image_dtype, resize)
from tensorflow import reshape
import tensorflow as tf
import json

def read_json(jsonPath):
	with open(jsonPath, "r") as f:
		data = json.load(f)
	return data

def get_image_c2w(jsonData, datasetPath):
	imagePaths = []

	c2ws = []  # Pose Matrics -> Camera to World Matrices

	for frame in jsonData["frames"]:
		imagePath = frame["file_path"]
		imagePath = imagePath.replace(".", datasetPath)
		imagePaths.append(f"{imagePath}.png")
		c2ws.append(frame["transform_matrix"])
	return (imagePaths, c2ws)


class GetImages():
	def __init__(self, imageWidth, imageHeight):
		self.imageWidth = imageWidth
		self.imageHeight = imageHeight

	def __call__(self, imagePath):
		image = read_file(imagePath)
		image = decode_jpeg(image, 3)
		image = convert_image_dtype(image, dtype=tf.float32)
		image = resize(image, (self.imageWidth, self.imageHeight))
		image = reshape(image, (self.imageWidth, self.imageHeight, 3))
		return image


# Directly from the NERF Paper Repository
class GetRays:	
	def __init__(self, focalLength, imageWidth, imageHeight, NEAR_BOUNDS, 
		FAR_BOUNDS, nC):
		self.focalLength = focalLength
		self.imageWidth = imageWidth
		self.imageHeight = imageHeight
		self.NEAR_BOUNDS = NEAR_BOUNDS
		self.FAR_BOUNDS = FAR_BOUNDS
		self.nC = nC

	def __call__(self, camera2world):
		# Creating a meshgrid for rays
		(x, y) = tf.meshgrid(
			tf.range(self.imageWidth, dtype=tf.float32),
			tf.range(self.imageHeight, dtype=tf.float32),
			indexing="xy",
		)

		# Define the camera coordinates
		xCamera = (x - self.imageWidth * 0.5) / self.focalLength
		yCamera = (y - self.imageHeight * 0.5) / self.focalLength

		# Define the camera vector
		xCyCzC = tf.stack([xCamera, -yCamera, -tf.ones_like(x)],
			axis=-1)

		# Slice the camera2world matrix to obtain the roataion and
		# Translation matrix
		rotation = camera2world[:3, :3]
		translation = camera2world[:3, -1]

		# Expand the camera coordinates to 
		xCyCzC = xCyCzC[..., None, :]
		
		# Get the world coordinates
		xWyWzW = xCyCzC * rotation
		
		# Calculate the direciton vector of the ray
		rayD = tf.reduce_sum(xWyWzW, axis=-1)
		rayD = rayD / tf.norm(rayD, axis=-1, keepdims=True)

		# Calculate the origin vector of the ray
		rayO = tf.broadcast_to(translation, tf.shape(rayD))

		# Get the sample points from the ray
		tVals = tf.linspace(self.NEAR_BOUNDS, self.FAR_BOUNDS, self.nC)
		noiseShape = list(rayO.shape[:-1]) + [self.nC]
		noise = (tf.random.uniform(shape=noiseShape) * 
			(self.FAR_BOUNDS - self.NEAR_BOUNDS) / self.nC)
		tVals = tVals + noise

		# Return origin, direction of the ray and the sample points
		return (rayO, rayD, tVals)