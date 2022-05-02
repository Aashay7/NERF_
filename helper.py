import torch 
from utils import img2mse, mse2psnr, encode_position, render_image_depth, sample_pdf, get_device
import config
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F


def eval_model( dataloader, coarseModel, fineModel, loss_fn):
    device = get_device()
    running_loss = 0.0
    with torch.no_grad():
        for image, originVectorCoarse, directionVectorCoarse, tValsCoarse in dataloader:
        
            image = torch.permute(image, (0,2,3,1))
            image = image / 255
            
            # r = o + t*d 
            raysCoarse = (originVectorCoarse[..., None, :] + 
                (directionVectorCoarse[..., None, :] * tValsCoarse[..., None]))
            
            
            #Encoding Inputs
            raysCoarse = encode_position(raysCoarse, config.xyzDims)
            dirsCoarse = torch.broadcast_to(directionVectorCoarse[..., None, :], size=tuple(raysCoarse[..., :3].size()))
            dirsCoarse = encode_position(dirsCoarse, config.dirDims)     
            
            # Sets model to TRAIN mode
            coarseModel.eval()
            (rgbCoarse, sigmaCoarse) = coarseModel(raysCoarse.to(device), dirsCoarse.to(device))
            
            # Sets Them to cpu        
            rgbCoarse = rgbCoarse.to('cpu')
            sigmaCoarse = sigmaCoarse.to('cpu')
            
            (renderedCoarseImage, renderedCoarseDepth, coarseWeights) = render_image_depth(rgb = rgbCoarse, sigma=sigmaCoarse, tVals= tValsCoarse)
            
            if fineModel != None:
                # compute the mid-points of t vals
                tValsCoarseMid = ((tValsCoarse[..., 1:] + tValsCoarse[..., :-1]) / 2.0)
        
                #Applying hierarchical sampling to get Fine points
                tValsFine = sample_pdf(tValsMid=tValsCoarseMid,
                    weights=coarseWeights, N=config.numberFine, batchSize=config.BATCH_SIZE, 
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
                fineModel.eval()
                (rgbFine, sigmaFine) = fineModel(raysFine.to(device), dirsFine.to(device))
                
                # Sets Them to cpu        
                rgbFine = rgbFine.to('cpu')
                sigmaFine = sigmaFine.to('cpu')
                
                (renderedFineImage, renderedFineDepth, FineWeights) = render_image_depth(rgb = rgbFine, 
                                                                                        sigma = sigmaFine, 
                                                                                        tVals = tValsFine)
                        
            #Calculate Coarse Loss
            loss = loss_fn(renderedCoarseImage, image)
            if fineModel != None:
                #Calculate Fine Loss
                loss = loss + loss_fn(renderedFineImage, image)
                    
            running_loss = running_loss + loss.item()
    return running_loss / len(dataloader)



def visualize_performance( epoch, image, originVectorCoarse, directionVectorCoarse, 
                          tValsCoarse, coarseModel, fineModel, valLossData, trainLossData, dir):
    device = get_device()
    
    with torch.no_grad():
        image = torch.permute(image, (0,2,3,1))
        image = image / 255
        
        # r = o + t*d 
        raysCoarse = (originVectorCoarse[..., None, :] + 
            (directionVectorCoarse[..., None, :] * tValsCoarse[..., None]))
        
        
        #Encoding Inputs
        raysCoarse = encode_position(raysCoarse, config.xyzDims)
        dirsCoarse = torch.broadcast_to(directionVectorCoarse[..., None, :], size=tuple(raysCoarse[..., :3].size()))
        dirsCoarse = encode_position(dirsCoarse, config.dirDims)     
                
        # Sets model to eval mode
        coarseModel.eval()
        (rgbCoarse, sigmaCoarse) = coarseModel(raysCoarse.to(device), dirsCoarse.to(device))
        
        # Sets Them to cpu        
        rgbCoarse = rgbCoarse.to('cpu')
        sigmaCoarse = sigmaCoarse.to('cpu')
        
        (renderedCoarseImage, renderedCoarseDepth, coarseWeights) = render_image_depth(rgb = rgbCoarse, sigma=sigmaCoarse, tVals= tValsCoarse)
        if fineModel != None: 
            # compute the mid-points of t vals
            tValsCoarseMid = ((tValsCoarse[..., 1:] + tValsCoarse[..., :-1]) / 2.0)

            #Applying hierarchical sampling to get Fine points
            tValsFine = sample_pdf(tValsMid=tValsCoarseMid,
                weights=coarseWeights, N=config.numberFine, batchSize=config.BATCH_SIZE, 
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
            fineModel.eval()
            (rgbFine, sigmaFine) = fineModel(raysFine.to(device), dirsFine.to(device))
            
            # Sets Them to cpu        
            rgbFine = rgbFine.to('cpu')
            sigmaFine = sigmaFine.to('cpu')
            
            (renderedFineImage, renderedFineDepth, FineWeights) = render_image_depth(rgb = rgbFine, 
                                                                                    sigma = sigmaFine, 
                                                                                    tVals = tValsFine)
                              
        # Converting output tensors to images
        image = np.asarray(image[0].detach())
        renderedCoarseImage = np.asarray(renderedCoarseImage[0].detach())
        renderedCoarseDepth = np.asarray(renderedCoarseDepth[0].detach())
        if fineModel:
            renderedFineImage   = np.asarray(renderedFineImage[0].detach())
            renderedFineDepth   = np.asarray(renderedFineDepth[0].detach())
        # Plot the rgb, depth and the loss plot.
        fig, ax = plt.subplots(nrows=2 if fineModel else 1, ncols=4, figsize=(25, 10))
               
        if fineModel:
            ax[0][0].imshow(image)
            ax[0][0].set_title(f"Actual Image: {epoch:03d}")
            
            ax[0][1].imshow(renderedCoarseImage)
            ax[0][1].set_title(f"Predicted Coarse Image: {epoch:03d}")

            ax[0][2].imshow(renderedCoarseDepth)
            ax[0][2].set_title(f"Depth Coarse Map: {epoch:03d}")

            ax[0][3].plot(valLossData)
            ax[0][3].set_xticks(np.arange(0, epoch + 1, 5.0))
            ax[0][3].set_title(f"Val Loss Plot: {epoch:03d}")
            
            ax[1][0].imshow(image)
            ax[1][0].set_title(f"Actual Image: {epoch:03d}")
            
            ax[1][1].imshow(renderedFineImage)
            ax[1][1].set_title(f"Predicted Fine Image: {epoch:03d}")

            ax[1][2].imshow(renderedFineDepth)
            ax[1][2].set_title(f"Depth Fine Map: {epoch:03d}")

            ax[1][3].plot(trainLossData)
            ax[1][3].set_xticks(np.arange(0, epoch + 1, 5.0))
            ax[1][3].set_title(f"Train Loss Plot: {epoch:03d}")
        else:
            ax[0].imshow(image)
            ax[0].set_title(f"Actual Image: {epoch:03d}")
            
            ax[1].imshow(renderedCoarseImage)
            ax[1].set_title(f"Predicted Coarse Image: {epoch:03d}")

            ax[2].imshow(renderedCoarseDepth)
            ax[2].set_title(f"Depth Coarse Map: {epoch:03d}")

            ax[3].plot(valLossData)
            ax[3].set_xticks(np.arange(0, epoch + 1, 5.0))
            ax[3].set_title(f"Val Loss Plot: {epoch:03d}")

        fig.savefig(f"{dir}/{epoch:03d}.png")
        plt.close()
