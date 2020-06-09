# Model
- Experiment with n1 and f1
- Change Padding. (From same to valid)
- compare result with official matlab code
- Search papers on SR using CNN
- Check initializers
- Change train img size from 20x20

# Improve Model
- LeakyReLU
- Pooling layer
- Keep padding same

# Result Visualization
- Get result of intermediate layers
- Crop
- Picture in picture (zoomed in part)
```
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    img = image[border: -border, border: -border]
    return img
```
