# Pre-process BDD dataset: https://deepdrive.berkeley.edu/
# Segmentation mask code for "Void" updated from 255 to 19 for continuous sequence
# This script must only be run once on downloaded dataset

from pathlib import Path
from tqdm import tqdm
from fastai.vision import get_image_files
import PIL

# Load label images
path_data = Path('../data/bdd100k/seg')
path_lbl = path_data / 'labels'
lbl_names = get_image_files(path_lbl, recurse=True)

# Replace "Void" mask from 255 to 19 (for better data display)
# TODO: Process should be *much* faster with cv2 but had issues with conversion
for lbl_name in tqdm(lbl_names):
    img = PIL.Image.open(lbl_name)
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixels[i, j] == 255:
                pixels[i, j] = 19
    img.save(lbl_name)