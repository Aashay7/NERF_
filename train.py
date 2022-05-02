import tensorflow as tf

# Set Seed for reproducibility
tf.random.set_seed(42)

# import the necessary packages
from NERF_utils.data import (read_json, get_image_c2w, GetImages, GetRays)
from NERF_utils.utils import (get_focal_from_fov, render_image_depth, sample_pdf)
from NERF_utils.encoder import encoder_fn
from NERF_utils.nerf import get_model
from NERF_utils.nerf_trainer import Nerf_Trainer
from NERF_utils.train_monitor import get_train_monitor
from NERF_utils import config
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os


## DATASET PREPARATION
print("Reading the data...")
traindata = read_json(config.TRAIN_DATA)
valdata = read_json(config.VAL_DATA)
testdata = read_json(config.TEST_DATA)

focalLength = get_focal_from_fov(fieldOfView=traindata["camera_angle_x"], width=config.IMAGE_WIDTH)

# print the focal length of the camera
print(f"[DETAIL] focal length of the camera: {focalLength}...")

# get the train, validation, and test image paths and camera2world
# matrices
print("[INIT - DATA LOAD] grabbing the image paths and camera2world matrices...")
trainImagePaths, trainC2Ws = get_image_c2w(jsonData=traindata, datasetPath=config.DATASET_PATH)
valImagePaths, valC2Ws = get_image_c2w(jsonData=valdata, datasetPath=config.DATASET_PATH)
testImagePaths, testC2Ws = get_image_c2w(jsonData=testdata, datasetPath=config.DATASET_PATH)

# instantiate a object of our class used to load images from disk
getImages = GetImages(imageHeight=config.IMAGE_HEIGHT, imageWidth=config.IMAGE_WIDTH)

# get the train, validation, and test image dataset
print("[UPDATE] building the image dataset pipeline...")

trainImageDs = (tf.data.Dataset.from_tensor_slices(trainImagePaths).map(getImages, num_parallel_calls=config.AUTO))
valImageDs   = (tf.data.Dataset.from_tensor_slices(valImagePaths).map(getImages, num_parallel_calls=config.AUTO))
testImageDs  = (tf.data.Dataset.from_tensor_slices(testImagePaths).map(getImages, num_parallel_calls=config.AUTO))

# instantiate the GetRays object
getRays = GetRays(focalLength=focalLength, imageWidth=config.IMAGE_WIDTH, imageHeight=config.IMAGE_HEIGHT, NEAR_BOUNDS=config.NEAR_BOUNDS, FAR_BOUNDS=config.FAR_BOUNDS, nC=config.NUM_COARSE)

# get the train validation and test rays dataset
print("[UPDATE] building the rays dataset pipeline...")
trainRayDs = (tf.data.Dataset.from_tensor_slices(trainC2Ws).map(getRays, num_parallel_calls=config.AUTO))
valRayDs = (tf.data.Dataset.from_tensor_slices(valC2Ws).map(getRays, num_parallel_calls=config.AUTO))
testRayDs = (tf.data.Dataset.from_tensor_slices(testC2Ws).map(getRays, num_parallel_calls=config.AUTO))

# zip the images and rays dataset together
traindata = tf.data.Dataset.zip((trainRayDs, trainImageDs))
valdata = tf.data.Dataset.zip((valRayDs, valImageDs))
testdata = tf.data.Dataset.zip((testRayDs, testImageDs))

# build data input pipeline for train, val, and test datasets
traindata = (
	traindata
	.shuffle(config.BATCH_SIZE)
	.batch(config.BATCH_SIZE)
	.repeat()
	.prefetch(config.AUTO)
)
valdata = (
	valdata
	.shuffle(config.BATCH_SIZE)
	.batch(config.BATCH_SIZE)
	.repeat()
	.prefetch(config.AUTO)
)
testdata = (
	testdata
	.batch(config.BATCH_SIZE)
	.prefetch(config.AUTO)
)

# Instantiation of the Coarse Model
coarseModel = get_model(lxyz=config.DIMS_XYZ, lDir=config.DIMS_DIR, batchSize=config.BATCH_SIZE, denseUnits=config.UNITS, skipLayer=config.SKIP_LAYER)

# Instantiation of the Fine Model
fineModel = get_model(lxyz=config.DIMS_XYZ, lDir=config.DIMS_DIR, batchSize=config.BATCH_SIZE, denseUnits=config.UNITS, skipLayer=config.SKIP_LAYER)

# NerF trainer model.
nerfModel = Nerf_Trainer(coarseModel=coarseModel, fineModel=fineModel, lxyz=config.DIMS_XYZ, lDir=config.DIMS_DIR, encoderFn=encoder_fn, renderImageDepth=render_image_depth, samplePdf=sample_pdf, nF=config.NUM_FINE)

# Compiling the Model (optimizer used : Adam, Loss Function : Mean Squared Error)
nerfModel.compile(optimizerCoarse=Adam(),optimizerFine=Adam(),lossFn=MeanSquaredError())

# Create an Image directory if not present.
if not os.path.exists(config.IMAGE_PATH):
	os.makedirs(config.IMAGE_PATH)

# Train Monitor Callback to track the model learning and evaluation results.
trainMonitorCallback = get_train_monitor(testDs=testdata,encoderFn=encoder_fn, lxyz=config.DIMS_XYZ, lDir=config.DIMS_DIR, imagePath=config.IMAGE_PATH)

# NERF Model training
print("Training initiated....")
nerfModel.fit(traindata, steps_per_epoch=config.STEPS_PER_EPOCH, validation_data=valdata, validation_steps=config.VALIDATION_STEPS, epochs=config.EPOCHS, callbacks=[trainMonitorCallback],)

# Saving the model parameters
nerfModel.coarseModel.save(config.COARSE_PATH)
nerfModel.fineModel.save(config.FINE_PATH)
print("Model parameters saved.")