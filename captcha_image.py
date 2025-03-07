import numpy as np
from PIL import Image

def normalize_image_to_np_array(path, size):
    image = Image.open(path).convert("L")  # gray style
    image = image.point(lambda p: 255 if p > 128 else 0) # binarization
    image = image.resize(size)  # unify size
    image_array = np.array(image) / 255.0  # normalization
    return image_array