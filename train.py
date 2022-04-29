from calendar import EPOCH
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from data import readJson, getImagePathsAndTMs, TotalSceneData
from utils import get_focal, get_device, img2mse, mse2psnr, encode_position, render_image_depth
from nerf import NerfModel
from skimage.io import imread
import config

#Constants
BASE_DIR = r"./data"
BATCH_SIZE = 5
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

nerf = NerfModel(numLayers = config.numLayers, xyzDims = config.xyzDims, 
                 dirDims = config.dirDims, batchSize = BATCH_SIZE, 
                 skipLayer = config.skipLayer, linearUnits = config.linearUnits)

device = get_device()

nerf.to(device)

#Intitializing the optimizer

optimizer = Adam(params=nerf.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))




train_loss = []
val_loss = []
for epoch in range(NUM_EPOCHS):
    train_running_loss = 0.0
    test_running_loss  = 0.0
    
    for image, originVector, directionVector, tVals in trainDataloader:
        
        image = torch.permute(image, (0,2,3,1))
        
        # r = o + t*d 
        raysCoarse = (originVector[..., None, :] + 
			(directionVector[..., None, :] * tVals[..., None]))
        
        
        #Encoding Inputs
        rays = encode_position(raysCoarse, config.xyzDims)
        dirs = torch.broadcast_to(directionVector[..., None, :], size=tuple(rays[..., :3].size()))
        dirs = encode_position(dirs, config.dirDims)     
        
        # Sets model to TRAIN mode
        nerf.train()
        (rgb, sigma) = nerf(rays, dirs)
        
        (renderedImage, renderedDepth, renderedWeight) = render_image_depth(rgb = rgb, sigma=sigma, tVals= tVals)
        print("Image shape ", image.size())
        print("renderedimage ", renderedImage.size())
        print("rendereddepth ", renderedDepth.size())
        print("renderedWeight ", renderedWeight.size())    
        
        #Calculate Loss
        loss = img2mse(renderedImage, image)
        #Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
        
    
    