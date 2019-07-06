import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
from functools import reduce
from scipy.spatial import ConvexHull

img = cv2.imread(r'D:\Petnica projekat\edge detection\1019 canny.jpg')
height = np.size(img, 0)
width = np.size(img, 1)
print(height,width)

#px=img[55,70]
#if px is [255,255,255]:
#print(px)

points = []
for i in range(height):
    for j in range(width):
        px = img[i,j,0]
        if px == 255:
            points.append([i,j])  



hull = ConvexHull(points)
points = [2,3,3,3,5,6]
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')





"""
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()


def convex_hull_graham(points):
    '''
        Returns points on convex hull in CCW order according to Graham's scan algorithm. 
    '''
    
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])

    return l.extend(u[i] for i in range(1, len(u) - 1)) or l

hull = convex_hull_graham(points)




plt.plot(points[:,0], points[:,1], 'o')
#for simplex in hull.simplices:
    #plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()


img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
contours,hierarchy = cv2.findContours(thresh,2,1)
print(len(contours))
cnt = contours[0]

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)

cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""