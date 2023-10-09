import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model.features import FeatureExtractor
from utils.background import *
from utils.ssim import *
from utils.utils import *
from utils.hessian import *
from utils.gaussian import gaussian_kernel
from model.skip import skip

'''
  Requires to be set based on the input image propersties.
'''
BCK=1.5
STD=1.5
SPARSITY=5.0
CONTINUITY=8e-7

f_model = FeatureExtractor().cuda()
f_model.load_weights('weights/feature_extractor.pt')
f_model.eval()


input = "image.png"

BLUR_IMG_PATH='nobackground.png'

im = np.array(Image.open(input).convert('L'))
im = im/np.max(im)
print('Image loaded.')

im_base_shape = im.shape

q = 16
pad_max = max(im.shape)
pad_0 = (((pad_max//q + 1) * q) - im.shape[0])
pad_1 = (((pad_max//q + 1) * q) - im.shape[1])

im = np.pad(im, pad_width=((0,pad_0),(0,pad_1)), mode='edge')

background = background_estimation(im/BCK)
im_X = im - background

print('Background subtracted.')

Image.fromarray(im_X[:im_base_shape[0],:im_base_shape[1]]*255).convert('L').save(BLUR_IMG_PATH)

psf = gaussian_kernel(21, STD)
psf = psf/np.sum(psf)
psf_torch = grayscale_to_torch(psf).float().to(device)

PAD = psf_torch.shape[-1]//2

print('Running deconvolution.')

# RGB input (for feature extractor)
im = get_input_image(BLUR_IMG_PATH)

pad_0 = (((im.shape[1]//q + 1) * q) - im.shape[0])
pad_1 = (((im.shape[1]//q + 1) * q) - im.shape[1])
im = np.pad(im, pad_width=((0,pad_0),(0,pad_1),(0,0)), mode='edge')

im_torch = rgb_to_torch(im).float().to(device)

# Grayscale input (for self-supervised loss)
im_L = get_grayscale_image(BLUR_IMG_PATH)
im_L_torch = grayscale_to_torch(im_L).float().to(device)

# ==============================================================================

# returns the deconvolved features
with torch.no_grad():
    _, fs = f_model(im_torch, psf_torch)

fs = fs.detach().to(device)
fs = fs[:,:,:im_base_shape[0],:im_base_shape[1]]
fs = F.pad(fs, (PAD,PAD,PAD,PAD), 'reflect')

# ==============================================================================

s_model = skip(
    num_input_channels=16,
    num_output_channels=1,
    num_channels_down = [32, 32, 64, 64],
    num_channels_up   = [32, 32, 64, 64],
    num_channels_skip = [16, 16, 16, 16],
    upsample_mode='bilinear',
    need_sigmoid=True,
    need_bias=True,
    act_fun='LeakyReLU',
).to(device)

s_model.train()

# ==============================================================================

NUM_EPOCHS=1000

optimizer = torch.optim.NAdam(s_model.parameters(), lr=1e-2, weight_decay=1e-5)
ssim = SSIM().cuda()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[700, 800, 900], gamma=0.5)

for epoch in range(0, NUM_EPOCHS):

    optimizer.zero_grad()

    x = s_model(fs)
    x_conv = torch.conv2d(x, psf_torch, stride=1, padding=0, bias=None).cuda()

    loss = 1 - ssim(x_conv, im_L_torch) + (torch.sum(x)/(x.shape[-2] * x.shape[-1])) * SPARSITY + hessian_loss(x[0]) * CONTINUITY

    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch % 100 == 0) or (epoch == (NUM_EPOCHS - 1)):
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch, NUM_EPOCHS, loss.item()))

print('Deconvolution done.')
print('Saving results as:', 'cider-output.png')

im = x.detach().cpu().numpy()[0,0,PAD:-PAD,PAD:-PAD]
Image.fromarray(im*255).convert('L').save('cider-output.png')

# -- END THINGS --

print('Deconvolution OK.')

print('Input image:')
plt.figure(figsize=(10,10))
plt.imshow(np.array(Image.open(input)), cmap='gray')
plt.axis('off')
plt.show()

print('Deconvolved image:')
plt.figure(figsize=(10,10))
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.show()