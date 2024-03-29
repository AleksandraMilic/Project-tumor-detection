"""
=====================
GLCM Texture Features
=====================

This example illustrates texture classification using grey level
co-occurrence matrices (GLCMs). A GLCM is a histogram of co-occurring
greyscale values at a given offset over an image.

In this example, samples of two different textures are extracted from
an image: grassy areas and sky areas. For each patch, a GLCM with
a horizontal offset of 5 is computed. Next, two features of the
GLCM matrices are computed: dissimilarity and correlation. These are
plotted to illustrate that the classes form clusters in feature space.

In a typical classification problem, the final step (not included in
this example) would be to train a classifier, such as logistic
regression, to label image patches from new images.

"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
from PIL import Image


PATCH_SIZE = 21

# open the camera image
image = cv2.imread(r'D:\Project-tumor-detection\preprocesiranje\preprocessed-images-gamma\507-2.jpg', 0)
#cv2.imshow("i",image)
#cv2.waitKey(0) -----------D:\Project-tumor-detection\segmentacija\maske\canny-detector-femur\1266.jpg ne prikazuje se
print(image)

# select some patches from  area 1 of the image
locations_1 = [(474, 291), (440, 433), (466, 18), (462, 236)]
patches_1 = []

for loc in locations_1:
	#patch = image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE]
	patches_1.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])



#print("grass", patches_1)
#print("piece of patches", patches_1[0][0])


# select some patches from area 2 of the image
locations_2 = [(54, 48), (21, 233), (90, 380), (195, 330)]
patches_2 = []
for loc in locations_2:
	patches_2.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (patches_1 + patches_2):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
	print("true")
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
    vmin=0, vmax=255)
for (y, x) in locations_1:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in locations_2:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(patches_1)], ys[:len(patches_1)], 'go',
        label='Patch 1')
ax.plot(xs[len(patches_2):], ys[len(patches_2):], 'bo',
        label='Patch 2')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(patches_1):
    ax = fig.add_subplot(3, len(patches_1), len(patches_1)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)

    ax.set_xlabel('Patch 1 %d' % (i + 1))

for i, patch in enumerate(patches_2):
    ax = fig.add_subplot(3, len(patches_2), len(patches_2)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)

    ax.set_xlabel('Patch 2 %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
