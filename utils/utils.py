from PIL import Image
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# returns RGB image, because the feature extractor is trained on RGB
def get_input_image(path):
    return np.array(Image.open(path).convert('RGB'))


def get_grayscale_image(path):
    return np.array(Image.open(path).convert('L'))/255


def get_notransform_image(path):
    return np.array(Image.open(path))


def get_psf(path):
    psf = np.array(Image.open(path).convert('L'))
    return psf/np.sum(psf)


def rgb_to_torch(x):
    return torch.from_numpy(x.T).unsqueeze(0)


def grayscale_to_torch(x):
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


def np_normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))


def torch_normalize(x):
    return (x - torch.min(x))/(torch.max(x) - torch.min(x))