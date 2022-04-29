import json
from torch.utils.data import Dataset
import torch 
from torchvision.io import read_image, ImageReadMode
import numpy as np

def readJson(jsonPath):
	# open the json file
	with open(jsonPath, "r") as fp:
		# read the json data
		data = json.load(fp)
	
	# return the data
	return data

def getImagePathsAndTMs(jsonData, basePath):
    imagePaths = [] 
    TMs = [] 
    
    for frame in jsonData["frames"]:
        imagePaths.append(basePath + r"/" + frame["file_path"][2:] + ".png")
        TMs.append(frame["transform_matrix"])
        
    return imagePaths,TMs

class TotalSceneData(Dataset):
    """Scene dataset class"""

    def __init__(self, imagePaths, c2wTMs, imageWidth, imageHeight, focalLength, near, far, n, image_transform=None):
        """
        Args:
            root_dir (string): Path to the directory which contains all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imagePaths = imagePaths
        self.c2wTMs = c2wTMs
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.focalLength = focalLength
        self.near = near
        self.far = far
        self.n = n
        self.image_transform = image_transform

    def __len__(self):
        assert len(self.imagePaths) == len(self.c2wTMs)
        return len(self.imagePaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        fname = self.imagePaths[idx]
        c2wTM = torch.Tensor(self.c2wTMs[idx])
        
        #Processing the Image
        image = read_image(fname, mode = ImageReadMode.RGB)
        
        if self.image_transform:
            image = self.image_transform(image)

        
        rotationMatrix = c2wTM[:3, :3]
        translationMatrix = c2wTM[:3, -1]

        (x, y) = torch.meshgrid(
			torch.arange(start=0, end=self.imageWidth, dtype=torch.float32),
            torch.arange(start=0, end=self.imageHeight, dtype=torch.float32),
			indexing="xy",
		)
        
        x = x.t()
        y = y.t()
        
        #Converting into Camera Co-ordinates
        dirs = torch.stack([(x - self.imageWidth * 0.5) / self.focalLength, - (y - self.imageHeight * 0.5) / self.focalLength, -torch.ones_like(x)], -1)
        
        #Getting the dvector
        cameraCoords = dirs[..., np.newaxis, :] #Getting the Camera Co-ordinates
        directionVector =  torch.sum(cameraCoords * rotationMatrix, dim=-1) #Calculating Direction Vector R X Camera Co-ordinates
        directionVector = directionVector / torch.norm(directionVector, dim=-1, keepdim=True) # Converting Directoion Vector into Unit Direction Vector
        
        # calculate the origin vector of the ray
        originVector = translationMatrix.expand(directionVector.shape)
		
    
        # Sample points from the ray 
        # Here we Sample points at regular intervals and then add noise taken from Uniform distribution 
        # This is done for better regularization 
        tVals = torch.linspace(self.near, self.far, self.n)
        noise = torch.rand(size=list(originVector.shape[:-1]) + [self.n]) * ((self.far - self.near) / self.n)
        tVals = tVals + noise
        
        return (image, originVector, directionVector, tVals)


