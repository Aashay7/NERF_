from tensorflow.data import AUTOTUNE
import os

# define the dataset path
DATAPATH = "dataset"

# define the json paths
TRAIN_DATA = os.path.join(DATAPATH, "transforms_train.json")
VAL_DATA = os.path.join(DATAPATH, "transforms_val.json")
TEST_DATA = os.path.join(DATAPATH, "transforms_test.json")

#To optimize the Tensorflow Data Pipelines
AUTO = AUTOTUNE

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100

# define the number of samples for coarse and fine model
NUM_COARSE = 32
NUM_FINE = 64

# define the dimension for positional encoding
DIMS_XYZ = 8
DIMS_DIR = 4

# define the NEAR_BOUNDS and FAR_BOUNDS bounding values of the 3D scene
NEAR_BOUNDS = 2.0
FAR_BOUNDS = 6.0

BATCH_SIZE = 1

# define the number of dense units
UNITS = 128

# define the skip layer
SKIP_LAYER = 4

# define the model fit parameters
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 5
EPOCHS = 1000

# define a output image path
OUTPUT_PATH = "output"
IMAGE_PATH = os.path.join(OUTPUT_PATH, "images")
VIDEO_PATH = os.path.join(OUTPUT_PATH, "videos")


# define coarse and fine model paths
COARSE_PATH = os.path.join(OUTPUT_PATH, "coarse")
FINE_PATH = os.path.join(OUTPUT_PATH, "fine")