import matplotlib.pyplot as plt
import scipy.interpolate as inter
import numpy as np
import warnings




def curve_fit(pts2):
    #n = len(coord)
    warnings.simplefilter('ignore', np.RankWarning)
    x = []
    y = []

    for i in pts2:
        x.append(i[0])
        y.append(i[1])
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, 'ro', ms=5)
    plt.show()



    xmin, xmax = min(x), max(x) 
    ymin, ymax = min(y), max(y)

    n = len(x)-1
    plotpoints = 1000 #100

    k = 3 #

    knotspace = range(n+1)
    #knots = inter.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
    #print("knots", knots)
    #knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))
    #print("knots_full", knots_full)

    #tX = knots_full, x, k
    #tY = knots_full, y, k

    splineX = inter.UnivariateSpline(knotspace, x, k=k)
    splineY = inter.UnivariateSpline(knotspace, y, k=k)

    tP = np.linspace(0, n, plotpoints)


    xP = splineX(tP)
    yP = splineY(tP)

    #print('xP', xP)

    plt.plot(xP, yP, 'g', lw=5)

    plt.show()

    return xP, yP


if __name__ == "__main__":
    pts2 = [(404, 699), (393, 673), (344, 227), (351, 57), (51, 54), (400, 99), (93, 673), (34, 22), (51, 157), (551, 354)]

    curve_fit(pts2)


"""
warnings.simplefilter('ignore', np.RankWarning)
coord = [(404, 699), (393, 673), (344, 227), (351, 57), (51, 54)]
n = len(coord)

t1 = np.arange(0, n, 0.01)

x_coordinate = []
y_coordinate = []

for i in coord:
    x_coordinate.append(i[0])
    y_coordinate.append(i[1])

print(x_coordinate)
print(y_coordinate)

#y_coordinate = y_coordinate.sort(reverse = True)


plt.plot(x_coordinate, y_coordinate, 'ro', ms=5)

s1 = inter.InterpolatedUnivariateSpline(x_coordinate, y_coordinate)

print(s1)






#s1rev = inter.InterpolatedUnivariateSpline(x[::-1], y[::-1])

np.linspace(1, n, 100)

for t in t1:
        
    spl_1 = inter.UnivariateSpline(t, x_coordinate, k = 3)
    spl_2 = inter.UnivariateSpline(t, y_coordinate, k = 3)

    x = spl_1(t)
    y = spl_2(t)

    plt.plot(t, x, 'g', lw=3)
    plt.plot(t, y, 'g', lw=3)
    



x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)
plt.plot(x, y, 'ro', ms=5)
#plt.show()


spl = inter.UnivariateSpline(x, y)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, spl(xs), 'g', lw=3)
plt.show()


spl.set_smoothing_factor(0.5)
plt.plot(xs, spl(xs), 'b', lw=3)
plt.show()

print(np.random.randn(50))



# Define some points:
theta = np.linspace(-3, 2, 40)
points = np.vstack( (np.cos(theta), np.sin(theta)) ).T


#points = points + 0.05*np.random.randn(*points.shape)

points = [[304, 699], [393, 673], [444, 227], [451, 57], [451, 54]]#, (450, 53), (448, 52), (440, 50), (248, 50), (66, 54), (50, 57), (50, 86), (129, 310), (301, 699)]


#Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1])

# Build a list of the spline function, one for each dimension:
splines = [inter.UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]

# Computed the spline for the asked distances:
alpha = np.linspace(0, 1, 75)
points_fitted = np.vstack( spl(alpha) for spl in splines ).T

# Graph:
plt.plot(*points.T, 'ok', label='original points');
plt.plot(*points_fitted.T, '-r', label='fitted spline k=3, s=.2');
plt.axis('equal'); plt.legend(); plt.xlabel('x'); plt.ylabel('y');
plt.show()





n = len(a)


x_coordinate = []
y_coordinate = []

for i in a:
    x_coordinate.append(i[0])
    y_coordinate.append(i[1])
    
    
    
for t in np.arange(0, n, 100): #range(0, n, 0.01):
    
    spl_1 = inter.UnivariateSpline(t, x_coordinate, k = 2)
    spl_2 = inter.UnivariateSpline(t, y_coordinate, k = 2)

    x = spl_1(t)
    y = spl_2(t)

    plt.plot(t, x, 'g', lw=3)
    plt.plot(t, y, 'g', lw=3)
plt.show()




from scipy.interpolate import InterpolatedUnivariateSpline

a = [(304, 699), (393, 673), (444, 227), (451, 57), (451, 54)]#, (450, 53), (448, 52), (440, 50), (248, 50), (66, 54), (50, 57), (50, 86), (129, 310), (301, 699)]
n = len(a)

x_coordinate = []
y_coordinate = []

for i in a:
    x_coordinate.append(i[0])
    y_coordinate.append(i[1])

x_coordinate, y_coordinate = np.array(x_coordinate), np.array(y_coordinate)

#x_coordinate = np.linspace(-3, 3, 50)
#y_coordinate = np.exp(-x**2) + 0.1 * np.random.randn(50)

spl = InterpolatedUnivariateSpline(x_coordinate, y_coordinate)
plt.plot(x_coordinate, y_coordinate, 'ro', ms=5)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
plt.show()





x = np.array([13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
y = np.array([2.404070, 1.588134, 1.760112, 1.771360, 1.860087,
        1.955789, 1.910408, 1.655911, 1.778952, 2.624719,
        1.698099, 3.022607, 3.303135])

p = np.arange(1,13.01,0.1)
s1 = inter.InterpolatedUnivariateSpline (x, y)
s1rev = inter.InterpolatedUnivariateSpline (x[::-1], y[::-1])



    n = len(pts2)
    x_coordinate, y_coordinate = Get_X_Y_coordinate(pts2)
    plt.plot(x_coordinate, y_coordinate, 'ro', ms=5)
    #plt.show()

    #spl = UnivariateSpline(x_coordinate, x_coordinate)
    #xs = np.linspace(-3, 3, 1000)
    #plt.plot(xs, spl(xs), 'g', lw=3)
    #plt.show()


    
    
    
    ###########np.linspace(1, n, 100): # za iscrtavanje
    

    
    for t in arange(0, n, 0.01):
        
        spl_1 = UnivariateSpline(t, x_coordinate, k = 3)
        spl_2 = UnivariateSpline(t, y_coordinate, k = 3)

        x = spl_1(t)
        y = spl_2(t)

        plt.plot(t, x, 'g', lw=3)
        plt.plot(t, y, 'g', lw=3)
        
        clean_im[x][y] = 255

    plt.show()
    
    

    


    points = pts2

    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]




    # Build a list of the spline function, one for each dimension:
    splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, 75)
    points_fitted = np.vstack( spl(alpha) for spl in splines ).T

    # Graph:
    
    
    plt.axis([0, w, 0, h]) ###########

    plt.plot(*points.T, 'ok', label='original points');
    plt.plot(*points_fitted.T, '-r', label='fitted spline k=3, s=.2');
    plt.axis('equal'); plt.legend(); plt.xlabel('x'); plt.ylabel('y');
    plt.show()


    """




