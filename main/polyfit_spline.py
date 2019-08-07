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
    # plt.plot(x, y, 'ro', ms=5)
    # plt.show()



    #xmin, xmax = min(x), max(x) 
    #ymin, ymax = min(y), max(y)

    n = len(x)-1
    plotpoints = 10000000 #10000000

    k = 5 #3

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

    # plt.plot(xP, yP, 'g', lw=5)

    # plt.show()

    return xP, yP


if __name__ == "__main__":
    pts2 = [(404, 699), (393, 673), (344, 227), (351, 57), (51, 54), (400, 99), (93, 673), (34, 22), (51, 157), (551, 354)]

    xP, yP = curve_fit(pts2)
    print(xP)
    print(yP)



