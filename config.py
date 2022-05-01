
train_json = "./data/transforms_train.json"
test_json = "./data/transforms_test.json"
val_json = "./data/transforms_val.json"

#BATch_SIZE
BATCH_SIZE = 5

#Image Height and Width 
image_width = 100
image_height = 100

#bounds
near = 2
far = 6

# number of ray sample points for coarse and fine model
# numberCoarse = 16
# numberFine = 32
numberCoarse = 32
numberFine = 64

# number of linear layers
numLayers = 8

# number of linear units
linearUnits = 5

# skip layer
skipLayer = 4

# define the dimension for positional encoding
xyzDims = 8
dirDims = 4

#LR decay in 1000s
lrate_decay = 250
#Decay Rate
decay_rate = 0.1
 