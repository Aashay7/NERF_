import torch.nn as nn
import torch.nn.functional as F
import torch


class NerfModel(nn.Module):
    def __init__(self, numLayers, xyzDims, dirDims, batchSize, skipLayer, linearUnits):
        super(NerfModel, self).__init__()
        
        self.xyzDims = xyzDims
        self.dirDims = dirDims
        self.batchSize = batchSize
        self.skipLayer = skipLayer
        self.linearUnits = linearUnits
        
        input_ray_shape = 2 * 3 * self.xyzDims + 3
        input_dir_shape = 2 * 3 * self.dirDims + 3
        
        self.linearLayers = nn.ModuleList(
            [nn.Linear(input_ray_shape ,self.linearUnits)] + 
                [nn.Linear(self.linearUnits, self.linearUnits) if not (i % self.skipLayer == 0 and i > 0) else 
                 nn.Linear(self.linearUnits + input_ray_shape, self.linearUnits) for i in range(numLayers-1)])
        
        
        self.sigmaLayer = nn.Linear(self.linearUnits, 1)
                
        self.featureLayer = nn.Linear(self.linearUnits, self.linearUnits)
        self.lastLinearLayer = nn.Linear(self.linearUnits + input_dir_shape, self.linearUnits//2)
        
        self.rgbLayer = nn.Linear(self.linearUnits//2, 3)
    
    def forward(self, input_ray, input_dir ):
        
        x = input_ray
        
        for i, _ in enumerate(self.linearLayers):
            x = self.linearLayers[i](x)
            x = F.relu(x)
            if (i % self.skipLayer == 0) and (i > 0):
                x = torch.cat([input_ray, x], -1)
        
           
        sigma = F.relu(self.sigmaLayer(x))
               
        feature = self.featureLayer(x)
        feature = torch.cat([feature, input_dir], -1) # Residual Connection
            
        x = F.relu(self.lastLinearLayer(feature))
        
        # rgb = nn.Sigmoid(self.rgbLayer(x))
        rgb = torch.sigmoid(self.rgbLayer(x))
        
        return (rgb, sigma)
