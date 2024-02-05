import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

idx=0
mask_suffix='_mask'
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
images_dir = Path(dir_img)
mask_dir = Path(dir_mask)

ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
if not ids:
    raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')



name = ids[idx]
mask_file = list(mask_dir.glob(name + mask_suffix + '.*'))
img_file = list(images_dir.glob(name + '.*'))

assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'


print(len(img_file))
print(len(mask_file))

mask = load_image(mask_file[0])
img = load_image(img_file[0])
    
assert img.size == mask.size, \
    f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

print(img.size)
print(mask.size)

img_ar = np.asarray(img)
print(img_ar.size)
print(img_ar.shape)
print(img_ar.ndim)
