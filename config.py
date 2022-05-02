
train_json = "./drums/transforms_train.json"
test_json = "./drums/transforms_test.json"
val_json = "./drums/transforms_val.json"

#BATch_SIZE
BATCH_SIZE = 1

#Image Height and Width 
image_width = 100
image_height = 100

#bounds
near = 2.0
far = 6.0

# number of ray sample points for coarse and fine model
numberCoarse = 32
numberFine = 64

# number of linear layers
numLayers = 8

# number of linear units
linearUnits = 128

# skip layer
skipLayer = 4

# define the dimension for positional encoding
xyzDims = 8
dirDims = 4

#LR decay in 1000s
lrate_decay = 500
#Decay Rate
decay_rate = 0.1