from tensorflow.data import AUTOTUNE
import os

# define the dataset path
DATASET_PATH = "dataset"

# define the json paths
TRAIN_JSON = os.path.join(DATASET_PATH, "transforms_train.json")
VAL_JSON = os.path.join(DATASET_PATH, "transforms_val.json")
TEST_JSON = os.path.join(DATASET_PATH, "transforms_test.json")

# define TensorFlow AUTOTUNE
AUTO = AUTOTUNE

# define image dimensions
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100

# define the number of samples for coarse and fine model
N_C = 32
N_F = 64

# define the dimension for positional encoding
L_XYZ = 8
L_DIR = 4

# define the near and far bounding values of the 3D scene
NEAR = 2.0
FAR = 6.0

# define the batch size
BATCH_SIZE = 1

# define the number of dense units
DENSE_UNITS = 128

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

# define the parameters of the rendered video
SAMPLE_THETA_POINTS = 180
FPS = 30
QUALITY = 7
MACRO_BLOCK_SIZE = None

# define the inference video path
OUTPUT_VIDEO_PATH = os.path.join(VIDEO_PATH, "output.mp4")

# define coarse and fine model paths
COARSE_PATH = os.path.join(OUTPUT_PATH, "coarse")
FINE_PATH = os.path.join(OUTPUT_PATH, "fine")