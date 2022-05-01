import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from data import readJson, getImagePathsAndTMs, TotalSceneData
from utils import get_focal, get_device, img2mse, mse2psnr, encode_position, render_image_depth, sample_pdf, make_dir, create_gif
from nerf import NerfModel
from helper import eval_model, visualize_performance
import config

#Constants
BASE_DIR = r"./drums"
CHECKPOINT_DIR = r"./Checkpoints"
IMAGE_DIR = r"./images"
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4

#Create Folders
make_dir(folderName = CHECKPOINT_DIR)
make_dir(folderName = IMAGE_DIR)

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
print("Using Device: ", device)
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
    print(f"Starting {epoch}...")
    train_running_loss = 0.0
    
    for image, originVectorCoarse, directionVectorCoarse, tValsCoarse in trainDataloader:
        # image = image.to(device)
        # originVectorCoarse = originVectorCoarse.to(device)
        # directionVectorCoarse = directionVectorCoarse.to(device)
        # tValsCoarse = tValsCoarse.to(device)
        
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
        (rgbCoarse, sigmaCoarse) = nerfCoarse(raysCoarse.to(device), dirsCoarse.to(device))
        
        # Sets Them to cpu        
        rgbCoarse = rgbCoarse.to('cpu')
        sigmaCoarse = sigmaCoarse.to('cpu')
        
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
        (rgbFine, sigmaFine) = nerfFine(raysFine.to(device), dirsFine.to(device))
        
        # Sets Them to cpu        
        rgbFine = rgbFine.to('cpu')
        sigmaFine = sigmaFine.to('cpu')
        
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
    
    train_loss.append(train_running_loss / len(trainDataloader))
    print(f"EPOCH {epoch} Training Completed, Current Training Loss: {train_loss[-1]}, Next Evaluating Model")
    
    # Evaluating on Val Data 
    eval_loss = eval_model(dataloader = valDataloader, coarseModel=nerfCoarse, fineModel=nerfFine)
    val_loss.append(eval_loss  / len(valDataloader))
    
    print(f"EPOCH {epoch} Evaluation Completed, Current Validation Loss: {val_loss[-1]}")
    # Apply ExponentialDecay to update LR
    newLR = LEARNING_RATE * (config.decay_rate ** (epoch / (config.lrate_decay * 1000)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = newLR
    
    image, originVectorCoarse, directionVectorCoarse, tValsCoarse = iter(testDataloader).next()
     
    visualize_performance(epoch = epoch, image = image, originVectorCoarse = originVectorCoarse, 
                          directionVectorCoarse = directionVectorCoarse, tValsCoarse = tValsCoarse, 
                          coarseModel = nerfCoarse, fineModel = nerfFine, valLossData = val_loss, 
                          trainLossData=train_loss,  dir = IMAGE_DIR)
    
    torch.save({
        'epoch': epoch,
        'nerfCoarse': nerfCoarse.state_dict(),
        'nerfFine': nerfFine.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, f"{CHECKPOINT_DIR}/ckpt_{epoch:03d}..tar") 
    
create_gif("images/*.png", "training.gif")