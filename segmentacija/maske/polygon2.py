import matplotlib.pyplot as plt
import cv2
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert

# The original image is inverted as the object must be white.
image = cv2.imread(r'D:\Petnica projekat\edge detection\1019 canny.jpg')

chull = convex_hull_image(image) #invert(data.horse())

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].set_title('Original picture')
ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_axis_off()

ax[1].set_title('Transformed picture')
ax[1].imshow(chull, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_axis_off()

plt.tight_layout()
plt.show()

