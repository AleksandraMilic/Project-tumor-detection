import matplotlib.pyplot as plt
import cv2
import numpy as np
from roi_polygon import SetPixels
"""
def preprocess_array(arr):
    to_return = []

    for e in arr:
        to_return.append(tuple(e[0].tolist()))
    return to_return


def convex_hull(points):
    Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    
    ####points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull. 
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]

# Example: convex hull of a 10-by-10 grid.
#assert convex_hull([(i//10, i%10) for i in range(100)]) == [(0, 0), (9, 0), (9, 9), (0, 9)]

img = cv2.imread(r'D:\Project-tumor-detection\slike\edges\edge 714-2.jpg')
ret1, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
height = np.size(img, 0)
width = np.size(img, 1)
print(height,width)

#px=img[55,70]
#if px is [255,255,255]:
#print(px)

#points = []
#for i in range(height):
 #   for j in range(width):
  #      px = img[i,j,0]
   #     if px == 0:
    #        points.append((i,j))  

points_array, points, image = SetPixels(img)

#points = np.array(points) ########
hull = cv2.convexHull(points_array,returnPoints = False)

#print(points)
#hull = convex_hull(points)
#print(hull)

#hull = [(0,0), (100, 100), (0,0)]

#hull = preprocess_array(hull)




i = 0
while i < len(hull):
    cv2.line(image, hull[i-1], hull[i], (255,0,0), 1)
    i += 1

cv2.line(image, hull[i - 1], hull[0], (255,0,0), 1)

#for i in range(1, len(hull)):
 #   cv2.line(img,hull[i-1],hull[i],(255,0,0),1)

#cv2.line(img,hull[-1],hull[1],(255,0,0),1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    
    ####points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull. 
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]

# Example: convex hull of a 10-by-10 grid.
#assert convex_hull([(i//10, i%10) for i in range(100)]) == [(0, 0), (9, 0), (9, 9), (0, 9)]

img = cv2.imread(r'D:\Petnica projekat\edge detection\gsa 806.jpg')
height = np.size(img, 0)
width = np.size(img, 1)
print(height,width)

px=img[55,70]
#if px is [255,255,255]:
print(px)

points = []
for i in range(height):
    for j in range(width):
        px = img[i,j,0]
        if px == 0:
            points.append((i,j))  

hull = cv2.convexHull(points,returnPoints = False)

#print(points)
#hull = convex_hull(points)
#print(hull)

#hull = [(0,0), (100, 100), (0,0)]

for i in range(1, len(hull)):
    cv2.line(img,hull[i-1],hull[i],(255,0,0),1)

cv2.line(img,hull[-1],hull[1],(255,0,0),1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()