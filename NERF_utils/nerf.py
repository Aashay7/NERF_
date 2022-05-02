from tensorflow.keras.layers import (Dense, concatenate)
from tensorflow.keras import (Input, Model)

def get_model(lxyz, lDir, batchSize, denseUnits, skipLayer):
	rayInput = Input(shape=(None, None, None, 2 * 3 * lxyz + 3), batch_size=batchSize)
	dirInput = Input(shape=(None, None, None, 2 * 3 * lDir + 3), batch_size=batchSize)
	
	x = rayInput
	for i in range(8):
		x = Dense(units=denseUnits, activation="relu")(x)
		if i % skipLayer == 0 and i > 0:
			x = concatenate([x, rayInput], axis=-1)
	
	sigma = Dense(units=1, activation="relu")(x)
	feature = Dense(units=denseUnits)(x)

	feature = concatenate([feature, dirInput], axis=-1)

	x = Dense(units=denseUnits//2, activation="relu")(feature)

	rgb = Dense(units=3, activation="sigmoid")(x)
	nerfModel = Model(inputs=[rayInput, dirInput], outputs=[rgb, sigma])
	return nerfModel