import numpy as np
import pandas as pd
import numba
import geostats

def variogram(x,y):
    return 0


def main():



    v = geostats.ExpVario(contribution=1.0, a=1.5, bearing=60-90, anisotropy=1.)
    gs = geostats.GeoStruct(variograms=v, nugget=0.0)


    n = 100
    x = np.arange(0, n, 1)
    y = np.arange(0, n, 1)

    X, Y = np.meshgrid(x,y)

    pts = np.vstack([X.ravel(), Y.ravel()])




    print('ehf')

if __name__ == '__main__':
    main()




