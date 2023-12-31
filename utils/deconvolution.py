import torch
import torch.nn as nn
import torch.nn.functional as F

def torch_richardson_lucy(image, psf, num_iter=50):
    """
    image: 4-dimensional input, NCHW format
    psf:   4-dimensional input, NCHW format
    """
    
    pad = (psf.shape[-1] + 1)

    image = image[:,:,0].unsqueeze(0).unsqueeze(0)
    image = (image - torch.min(image))/(torch.max(image) - torch.min(image))

    image = torch.nn.functional.pad(image, (pad,pad,pad,pad), mode='reflect')

    im_deconv = torch.full(image.shape, 0.5).cuda()
    psf_mirror = torch.flip(psf, (-2,-1))

    eps = 1e-12

    for _ in range(num_iter):
        conv = torch.conv2d(im_deconv, psf, stride=1, padding=psf.shape[-1]//2) + eps
        relative_blur = image / conv
        im_deconv *= torch.conv2d(relative_blur, psf_mirror, stride=1, padding=psf.shape[-1]//2) + eps    
        im_deconv = torch.clip(im_deconv, -1, 1)
    
    return im_deconv[:,:,pad:-pad,pad:-pad]