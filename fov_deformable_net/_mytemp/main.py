import cv2
import numpy as np
from numpy.random import RandomState
from PIL import Image

from degradation import ImageDegradation


def main():
    import os
    from glob import glob
    from tqdm import tqdm

    SRC_IMAGE_DIR = "/datasets/arglass/eye_data/"
    SHARP_IMAGE_DIR = "/datasets/arglass/mydata/val/gt/"
    BLURRED_IMAGE_DIR = "/datasets/arglass/mydata/val/input/"

    rd = RandomState(42)
    degradation_model = ImageDegradation(rd, min_defocus_kernel = (4, 5), max_defocus_kernel = (12, 13), noise_poisson = None, noise_gaussian = (0.01,0.03))

    os.makedirs(SHARP_IMAGE_DIR, mode=0o755, exist_ok="True")    
    os.makedirs(BLURRED_IMAGE_DIR, mode=0o755, exist_ok="True")    

    from lykutils.baseutils import temporary_seed
    import random

    if False:
        with temporary_seed(0):
            files = glob(os.path.join(SRC_IMAGE_DIR, "**/*.png"), recursive=True)
            random.shuffle(files)
            files = files[:10000]
    else:
        with open("/datasets/arglass/val_datalist.txt", "r") as f:
            files = f.read().splitlines()

    for i, file in tqdm(list(enumerate(files))):
        file = file.replace("./datas/", SRC_IMAGE_DIR)

        sharp = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        sharp_pad = cv2.copyMakeBorder(sharp, 13, 13, 13, 13, cv2.BORDER_REPLICATE)
        sharp_pad = sharp_pad.astype(np.float32) / 255

        sharp = Image.fromarray(sharp)
        blurred = degradation_model.apply(sharp_pad)
        blurred = Image.fromarray(np.clip(blurred * 255, 0, 255).astype(np.uint8))
        
        filename = str(i).zfill(6) + ".png"
        sharp.save(os.path.join(SHARP_IMAGE_DIR, filename))
        blurred.save(os.path.join(BLURRED_IMAGE_DIR, filename))

    print("Done")


if __name__ == "__main__":
    main()