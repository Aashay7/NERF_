from calendar import EPOCH
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from data import readJson, getImagePathsAndTMs, TotalSceneData
from utils import get_focal, get_device, img2mse, mse2psnr, encode_position, render_image_depth, sample_pdf
from nerf import NerfModel
from helper import eval_model
import config

#Constants
BASE_DIR = r"./data"
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = 1
LEARNING_RATE = 5e-4

print("Reading the Json Data")

#Reading the Json Data 
trainData = readJson(config.train_json)
testData = readJson(config.test_json)
valData = readJson(config.val_json)

#GetFocalLength
focal = get_focal(camera_angle= trainData["camera_angle_x"],
                  width= config.image_width)

print(f"Focal length: {focal}")

trainImagePaths, trainC2WTMs = getImagePathsAndTMs(jsonData=trainData, basePath=BASE_DIR)
testImagePaths, testC2WTMs = getImagePathsAndTMs(jsonData=testData, basePath=BASE_DIR)
valImagePaths, valC2WTMs = getImagePathsAndTMs(jsonData=valData, basePath=BASE_DIR)


#Transforms for the Images
data_transforms = transforms.Compose([
    transforms.Resize((config.image_height, config.image_width))
])


#Train, Test and Val Data loaders
trainDataset = TotalSceneData(imagePaths= trainImagePaths, c2wTMs = trainC2WTMs, 
                                    focalLength=focal, imageWidth  = config.image_width, imageHeight = config.image_height, 
                                    near = config.near, far = config.far, n=config.numberCoarse, image_transform = data_transforms)

trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)

testDataset = TotalSceneData(imagePaths= testImagePaths, c2wTMs = testC2WTMs, 
                                    focalLength=focal, imageWidth  = config.image_width, imageHeight = config.image_height, 
                                    near = config.near, far = config.far, n=config.numberCoarse, image_transform = data_transforms)

testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)

valDataset = TotalSceneData(imagePaths= valImagePaths, c2wTMs = valC2WTMs, 
                                    focalLength=focal, imageWidth  = config.image_width, imageHeight = config.image_height, 
                                    near = config.near, far = config.far, n=config.numberCoarse, image_transform = data_transforms)

valDataloader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True)

#Nerf model

nerfCoarse = NerfModel(numLayers = config.numLayers, xyzDims = config.xyzDims, 
                 dirDims = config.dirDims, batchSize = BATCH_SIZE, 
                 skipLayer = config.skipLayer, linearUnits = config.linearUnits)

nerfFine = NerfModel(numLayers = config.numLayers, xyzDims = config.xyzDims, 
                 dirDims = config.dirDims, batchSize = BATCH_SIZE, 
                 skipLayer = config.skipLayer, linearUnits = config.linearUnits)

device = get_device()

nerfCoarse.to(device)
nerfFine.to(device)
#Intitializing the optimizer

optimizer = Adam(params = list(nerfCoarse.parameters()) + list(nerfFine.parameters()), 
                 lr=LEARNING_RATE, betas=(0.9, 0.999))


train_loss = []
val_loss = []

# psnr_coarse_train_loss = []
# psnr_coarse_train_loss = []
# psnr_fine_val_loss = []
# psnr_fine_val_loss = []

for epoch in range(NUM_EPOCHS):
    train_running_loss = 0.0
    
    for image, originVectorCoarse, directionVectorCoarse, tValsCoarse in trainDataloader:
        
        image = torch.permute(image, (0,2,3,1))
        
        # r = o + t*d 
        raysCoarse = (originVectorCoarse[..., None, :] + 
			(directionVectorCoarse[..., None, :] * tValsCoarse[..., None]))
        
        
        #Encoding Inputs
        raysCoarse = encode_position(raysCoarse, config.xyzDims)
        dirsCoarse = torch.broadcast_to(directionVectorCoarse[..., None, :], size=tuple(raysCoarse[..., :3].size()))
        dirsCoarse = encode_position(dirsCoarse, config.dirDims)     
        
        # Sets model to TRAIN mode
        nerfCoarse.train()
        (rgbCoarse, sigmaCoarse) = nerfCoarse(raysCoarse, dirsCoarse)
        
        (renderedCoarseImage, renderedCoarseDepth, coarseWeights) = render_image_depth(rgb = rgbCoarse, sigma=sigmaCoarse, tVals= tValsCoarse)
        
        # compute the mid-points of t vals
        tValsCoarseMid = ((tValsCoarse[..., 1:] + tValsCoarse[..., :-1]) / 2.0)
  
        #Applying hierarchical sampling to get Fine points
        tValsFine = sample_pdf(tValsMid=tValsCoarseMid,
			weights=coarseWeights, N=config.numberFine, batchSize=BATCH_SIZE, 
            imageHeight = config.image_height, imageWidth = config.image_width)
        
        tValsFine, _ = torch.sort(
            torch.cat([tValsCoarse, tValsFine], -1 ), -1

        )
        
        raysFine = (originVectorCoarse[..., None, :] + 
			(directionVectorCoarse[..., None, :] * tValsFine[..., None]))
        raysFine = encode_position(raysFine, config.xyzDims)
        
        dirsFine = torch.broadcast_to(directionVectorCoarse[..., None, :], size=tuple(raysFine[..., :3].size()))
        dirsFine = encode_position(dirsFine, config.dirDims)
        
        # Sets model to TRAIN mode
        nerfFine.train()
        (rgbFine, sigmaFine) = nerfFine(raysFine, dirsFine)
        
        (renderedFineImage, renderedFineDepth, FineWeights) = render_image_depth(rgb = rgbFine, 
                                                                                 sigma = sigmaFine, 
                                                                                 tVals = tValsFine)
        #Optimization
        optimizer.zero_grad() 
        
        #Calculate Coarse Loss
        loss = img2mse(renderedCoarseImage, image)
        #Calculate Fine Loss and adding it with coarse loss
        loss = loss + img2mse(renderedFineImage, image)  
        
        #Backprop
        loss.backward()
        optimizer.step()
        
        train_running_loss = train_running_loss + loss.item()
        break
    
    train_loss.append(train_running_loss / len(trainDataloader))
    print("train_loss: ", train_loss)
    
    # Evaluating on Val Data 
    eval_loss = eval_model(dataloader = valDataloader, coarseModel=nerfCoarse, fineModel=nerfFine)
    val_loss.append(eval_loss)
    print("val_loss: ", val_loss)
    
    
    ###   update learning rate   ###
    decay_rate = 0.1
    decay_steps = args.lrate_decay * 1000
    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = 
    exit(1)