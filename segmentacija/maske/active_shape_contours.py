import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2

from polygon_correct2 import main


img = cv2.imread('D:\\Project-tumor-detection\\slike\\test\\edge-operators\\canny\\age 40, m.jpeg',0)
#img = rgb2gray(img)

# s = np.linspace(0, 2*np.pi, 400)
# x = 920 + 100*np.cos(s)
# y = 500 + 100*np.sin(s)
# init = np.array([x, y]).T
PATCH_SIZE = 50 #### 10
im_2, pts2, array_hull = main(PATCH_SIZE, img)
init = []
for i in array_hull:
    init.append([i[0],i[1]])
print(init)


snake = active_contour(gaussian(img, 3), init, alpha=0.015, beta=10, gamma=0.001)
print(snake)

# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# #ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
# ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])

# plt.show()