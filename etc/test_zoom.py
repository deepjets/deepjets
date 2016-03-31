from scipy.ndimage.interpolation import zoom
from skimage.transform import rescale, resize
import numpy as np


image = np.arange(9).reshape(3, 3).astype(np.float64)


def zoom(image, factor):
    w, h = image.shape
    new_w, new_h = np.ceil(factor * w), np.ceil(factor * h)
    if new_h % 2 != h % 2:
        new_h -= 1
    if new_w % 2 != w % 2:
        new_w -= 1
    scaled = resize(image, (new_w, new_h), order=1, mode='nearest')
    # crop
    pad_x = (new_w - w) / 2
    pad_y = (new_h - h) / 2
    return scaled[pad_x:new_w - pad_x, pad_y:new_h - pad_y]


print image
print image.shape
for factor in np.linspace(1, 100, 10):
    zoomed = zoom(image, factor=factor)
    print zoomed
    print zoomed.shape
