import cv2
import matplotlib.pyplot as plt
import numpy as np

def drawlines(img, points):
    points_array = np.array(points)
    #print(type(np.array(points)), "points type")
    #print(points.dtype, "dtype")
    #print(points.shape, "shape")
    filler = cv2.convexHull(points_array)
    print(filler)
    cv2.polylines(img, filler, True, (255, 255, 255), thickness=2)
    return img 

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
    l=lower[:-1] + upper[:-1]
    return [(y,x) for x,y in l]




# Make empty black image
w=400
h=400
image=np.zeros((h,w), np.uint8) #h,w,3

#points
pts = np.array([[50,70],[50,80],[70,70],[70,90],[50,90],[60,80],[65,85]], np.int32)

# Change pixels
image[50,70]=255
image[70,70]=255
image[70,90]=255
image[50,90]=255
image[60,80]=255
image[65,85]=255
image[50,80]=255

cv2.imshow('img1', image)


hull = convex_hull(pts)
for i in range(1, len(hull)):
    cv2.line(image,hull[i-1],hull[i],(255,0,0),1)

cv2.line(image,hull[-1],hull[0],(255,0,0),1)


#hull = cv2.convexHull(pts)
#img2 = cv2.fillConvexPoly(image, hull, (255,255,255), lineType=8, shift=0) 


cv2.imshow('img2', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Draw a point based on the x, y axis value.
# Draw a point at the location (3, 9) with size 1000


#plt.plot(points[:,0], points[:,1], 'o')
"""
pts = np.array([[3,9],[3,14],[7,9],[7,14],[4,11],[5,10]])

hull = convex_hull(pts)

plt.plot(pts[:,0], pts[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(pts[simplex, 0], pts[simplex, 1], 'k-')


plt.show()
"""


"""
from scipy.spatial import ConvexHull, convex_hull_plot_2d
points = np.array([[3,9],[3,14],[7,9],[7,14],[4,11],[5,10]])   # 30 random points in 2-D
hull = ConvexHull(points)

plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()"""