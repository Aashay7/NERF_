import os
import imageio
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from NERF_utils import config
from NERF_utils.utils import (pose_spherical, render_image_depth, sample_pdf, get_focal_from_fov)
from NERF_utils.data import (GetRays, read_json)
from NERF_utils.encoder import encoder_fn
from tensorflow.keras.models import load_model

c2wList = []

for theta in np.linspace(0.0, 360.0, config.SAMPLE_THETA_POINTS, endpoint=False):
	c2w = pose_spherical(theta, -30.0, 4.0)
	c2wList.append(c2w)

print("Reading the data...")
traindata = read_json(config.TRAIN_JSON)
focalLength = get_focal_from_fov(fieldOfView=traindata["camera_angle_x"], width=config.IMAGE_WIDTH)


getRays = GetRays(focalLength=focalLength, imageWidth=config.IMAGE_WIDTH, imageHeight=config.IMAGE_HEIGHT, near=config.NEAR, far=config.FAR, nC=config.N_C)

ds = (tf.data.Dataset.from_tensor_slices(c2wList).map(getRays).batch(config.BATCH_SIZE))

coarseModel = load_model(config.COARSE_PATH, compile=False)
fineModel = load_model(config.FINE_PATH, compile=False)

print("Generating the Novel Views...")
frameList = []
for element in tqdm(ds):
	(raysOriCoarse, raysDirCoarse, tValsCoarse) = element
	raysCoarse = (raysOriCoarse[..., None, :] + (raysDirCoarse[..., None, :] * tValsCoarse[..., None]))

	# Positional encoding the rays and the directions
	raysCoarse = encoder_fn(raysCoarse, config.L_XYZ)
	dirCoarseShape = tf.shape(raysCoarse[..., :3])
	dirsCoarse = tf.broadcast_to(raysDirCoarse[..., None, :], shape=dirCoarseShape)
	dirsCoarse = encoder_fn(dirsCoarse, config.L_DIR)

	(rgbCoarse, sigmaCoarse) = coarseModel.predict([raysCoarse, dirsCoarse])
	
	renderCoarse = render_image_depth(rgb=rgbCoarse, sigma=sigmaCoarse, tVals=tValsCoarse)
	(_, _, weightsCoarse) = renderCoarse

	tValsCoarseMid = (0.5 * (tValsCoarse[..., 1:] + tValsCoarse[..., :-1]))

	tValsFine = sample_pdf(tValsMid=tValsCoarseMid, weights=weightsCoarse, nF=config.N_F)
	tValsFine = tf.sort(tf.concat([tValsCoarse, tValsFine], axis=-1), axis=-1)

	raysFine = (raysOriCoarse[..., None, :] + (raysDirCoarse[..., None, :] * tValsFine[..., None]))
	raysFine = encoder_fn(raysFine, config.L_XYZ)
	
	dirsFineShape = tf.shape(raysFine[..., :3])
	dirsFine = tf.broadcast_to(raysDirCoarse[..., None, :], shape=dirsFineShape)
	dirsFine = encoder_fn(dirsFine, config.L_DIR)

	(rgbFine, sigmaFine) = fineModel.predict([raysFine, dirsFine])
	
	renderFine = render_image_depth(rgb=rgbFine, sigma=sigmaFine, tVals=tValsFine)
	(imageFine, _, _) = renderFine

	frameList.append(imageFine.numpy()[0])

if not os.path.exists(config.VIDEO_PATH):
	os.makedirs(config.VIDEO_PATH)

# Video Generation
print("Video Generation In Progress...")
imageio.mimwrite(config.OUTPUT_VIDEO_PATH, frameList, fps=config.FPS,
	quality=config.QUALITY, macro_block_size=config.MACRO_BLOCK_SIZE)
print("Video Generation Complete!")