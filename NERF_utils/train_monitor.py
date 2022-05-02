from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf

def get_train_monitor(testDs, encoderFn, lxyz, lDir, imagePath):
	(tElements, tImages) = next(iter(testDs))
	(tRaysOriCoarse, tRaysDirCoarse, tTvalsCoarse) = tElements

	tRaysCoarse = (tRaysOriCoarse[..., None, :] + (tRaysDirCoarse[..., None, :] * tTvalsCoarse[..., None]))

	# Coarse Ray -> Positional Encoding of the Coarse Ray and Direction vector
	tRaysCoarse = encoderFn(tRaysCoarse, lxyz)
	tDirsCoarseShape = tf.shape(tRaysCoarse[..., :3])
	tDirsCoarse = tf.broadcast_to(tRaysDirCoarse[..., None, :], shape=tDirsCoarseShape)
	tDirsCoarse = encoderFn(tDirsCoarse, lDir)

	class TrainMonitor(Callback):
		def on_epoch_end(self, epoch, logs=None):
			(tRgbCoarse, tSigmaCoarse) = self.model.coarseModel.predict([tRaysCoarse, tDirsCoarse])
			
			tRenderCoarse = self.model.renderImageDepth(rgb=tRgbCoarse, sigma=tSigmaCoarse, tVals=tTvalsCoarse)
			(tImageCoarse, _, tWeightsCoarse) = tRenderCoarse

			# Mid Values of the tvals
			tTvalsCoarseMid = (0.5 * (tTvalsCoarse[..., 1:] + tTvalsCoarse[..., :-1]))

			# Applying hierarchical sampling and get the t vals for the fine model
			tTvalsFine = self.model.samplePdf(tValsMid=tTvalsCoarseMid, weights=tWeightsCoarse, nF=self.model.nF)
			tTvalsFine = tf.sort(tf.concat([tTvalsCoarse, tTvalsFine], axis=-1),axis=-1)

			# Building the fine rays and encoding 
			tRaysFine = (tRaysOriCoarse[..., None, :] + (tRaysDirCoarse[..., None, :] * tTvalsFine[..., None]))
			tRaysFine = self.model.encoderFn(tRaysFine, lxyz)
			
			# Builiding the directions and encoding
			tDirsFineShape = tf.shape(tRaysFine[..., :3])
			tDirsFine = tf.broadcast_to(tRaysDirCoarse[..., None, :], shape=tDirsFineShape)
			tDirsFine = self.model.encoderFn(tDirsFine, lDir)

			# compute the fine model prediction
			tRgbFine, tSigmaFine = self.model.fineModel.predict([tRaysFine, tDirsFine])
			
			# render the image from the model prediction
			tRenderFine = self.model.renderImageDepth(rgb=tRgbFine, sigma=tSigmaFine, tVals=tTvalsFine)
			(tImageFine, tDepthFine, _) = tRenderFine

			# plot the coarse image, fine image, fine depth map and target image
			(_, ax) = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
			ax[0].imshow(array_to_img(tImageCoarse[0]))
			ax[0].set_title(f"Corase Image")

			ax[1].imshow(array_to_img(tImageFine[0]))
			ax[1].set_title(f"Fine Image")

			ax[2].imshow(array_to_img(tDepthFine[0, ..., None]), 
				cmap="inferno")
			ax[2].set_title(f"Fine Depth Image")

			ax[3].imshow(array_to_img(tImages[0]))
			ax[3].set_title(f"Real Image")

			plt.savefig(f"{imagePath}/{epoch:03d}.png")
			plt.close()
	
	trainMonitor = TrainMonitor()
	return trainMonitor