import torch 
import numpy as np


def cumprod_exclusive(tensor):
    cumprod = torch.cumprod(tensor, dim = -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def render_image_depth(rgb, sigma, tVals):
    
    sigma = torch.squeeze(sigma)
    
    diff = tVals[..., 1:] - tVals[..., :-1]
    
    diff = torch.cat([diff, torch.Tensor([1e10]).expand(diff[...,:1].shape)], -1)
    
    alpha = 1.0 - torch.exp( - sigma * diff)
        
    # transmittance  = cumprod_exclusive(1.0-alpha + 1e-10)
    transmittance  = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], alpha.shape[2], 1)), 1.0-alpha + 1e-10], -1), -1)[:, :, :, :-1]
    weights = alpha * transmittance
    
    image = torch.sum(weights[...,None] * rgb, -2)
    depth = torch.sum(weights * tVals, -1)
    
    # return rgb, depth map and weights
    return (image, depth, weights)

def encode_position(x, dims):
    
    positions = [x]
    for i in range(dims):
        for fn in [torch.sin , torch.cos]:
            positions.append(fn(2.0 ** i * x))
    
    return torch.concat(positions, axis=-1)


def sample_pdf(tValsMid, weights, N, batchSize, imageHeight, imageWidth):
	# add a small value to the weights to prevent it from nan
	weights = weights + 1e-5
	# normalize the weights to get the pdf
	pdf = weights / torch.sum(weights, dim=-1, keepdims=True)
	# from pdf to cdf transformation
	cdf = torch.cumsum(pdf, -1)
	# start the cdf with 0s
	cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
	# get the sample points
	u = torch.rand(list(cdf.shape[:-1]) + [N])
	u = u.contiguous()
	# get the indices of the points of u when u is inserted into cdf in a
	# sorted manner
	indices = torch.searchsorted(cdf, u, right=True)
	# define the boundaries
	below = torch.max(torch.zeros_like(indices -1), indices-1)
	above = torch.min((cdf.shape[-1]-1) * torch.zeros_like(indices), indices)	
	indicesG = torch.stack([below, above], -1)
	
	matched_shape = [indicesG.shape[0], indicesG.shape[1], indicesG.shape[2], indicesG.shape[3], cdf.shape[-1]]
	# gather the cdf according to the indices
	cdfG = torch.gather(cdf.unsqueeze(3).expand(matched_shape), 2, indicesG )
	
	matched_shape = [indicesG.shape[0], indicesG.shape[1], indicesG.shape[2], indicesG.shape[3], tValsMid.shape[-1]]
	# gather the tVals according to the indices
	tValsMidG = torch.gather(tValsMid.unsqueeze(3).expand(matched_shape), 2, indicesG)
	
    # create the samples by inverting the cdf
	denom = cdfG[..., 1] - cdfG[..., 0]
	denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
	t = (u - cdfG[..., 0]) / denom
	samples = (tValsMidG[..., 0] + t * 
		(tValsMidG[..., 1] - tValsMidG[..., 0]))
	
	# return the samples
	return samples


img2mse = lambda x, y : torch.mean((x - y) ** 2)

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def get_focal(camera_angle, width):
    print(camera_angle)
    print(width)
    return (0.5 * width) / (np.tan(0.5 * float(camera_angle)))