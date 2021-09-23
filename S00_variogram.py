import numpy as np
import pandas as pd
import numba

EPSILON = 1.0e-7

class Vario2d(object):
    """base class for 2-D variograms.
    Args:
        contribution (float): sill of the variogram
        a (`float`): (practical) range of correlation
        anisotropy (`float`, optional): Anisotropy ratio. Default is 1.0
        bearing : (`float`, optional): angle in degrees East of North corresponding
            to anisotropy ellipse. Default is 0.0
        name (`str`, optinoal): name of the variogram.  Default is "var1"
    Note:
        This base class should not be instantiated directly as it does not implement
        an h_function() method.
    """

    def __init__(self, contribution, a, anisotropy=1.0, bearing=0.0, name="var1"):
        self.name = name
        self.epsilon = EPSILON
        self.contribution = float(contribution)
        assert self.contribution > 0.0
        self.a = float(a)
        assert self.a > 0.0
        self.anisotropy = float(anisotropy)
        assert self.anisotropy > 0.0
        self.bearing = float(bearing)

    def same_as_other(self, other):
        if type(self) != type(other):
            return False
        if self.contribution != other.contribution:
            return False
        if self.anisotropy != other.anisotropy:
            return False
        if self.a != other.a:
            return False
        if self.bearing != other.bearing:
            return False
        return True

    def to_struct_file(self, f):
        """write the `Vario2d` to a PEST-style structure file
        Args:
            f (`str`): filename to write to.  `f` can also be an open
                file handle.
        """
        if isinstance(f, str):
            f = open(f, "w")
        f.write("VARIOGRAM {0}\n".format(self.name))
        f.write("  VARTYPE {0}\n".format(self.vartype))
        f.write("  A {0}\n".format(self.a))
        f.write("  ANISOTROPY {0}\n".format(self.anisotropy))
        f.write("  BEARING {0}\n".format(self.bearing))
        f.write("END VARIOGRAM\n\n")

    @property
    def bearing_rads(self):
        """get the bearing of the Vario2d in radians
        Returns:
            `float`: the Vario2d bearing in radians
        """
        return (np.pi / 180.0) * (90.0 - self.bearing)

    @property
    def rotation_coefs(self):
        """get the rotation coefficents in radians
        Returns:
            [`float`]: the rotation coefficients implied by `Vario2d.bearing`
        """
        return [
            np.cos(self.bearing_rads),
            np.sin(self.bearing_rads),
            -1.0 * np.sin(self.bearing_rads),
            np.cos(self.bearing_rads),
        ]

    def inv_h(self, h):
        """the inverse of the h_function.  Used for plotting
        Args:
            h (`float`): the value of h_function to invert
        Returns:
            `float`: the inverse of h
        """
        return self.contribution - self._h_function(h)

    def plot(self, **kwargs):
        """get a cheap plot of the Vario2d
        Args:
            **kwargs (`dict`): keyword arguments to use for plotting
        Returns:
            `matplotlib.pyplot.axis`
        Note:
            optional arguments in kwargs include
            "ax" (existing `matplotlib.pyplot.axis`).  Other
            kwargs are passed to `matplotlib.pyplot.plot()`
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise Exception("error importing matplotlib: {0}".format(str(e)))

        ax = kwargs.pop("ax", plt.subplot(111))
        x = np.linspace(0, self.a * 3, 100)
        y = self.inv_h(x)
        ax.set_xlabel("distance")
        ax.set_ylabel(r"$\gamma$")
        ax.plot(x, y, **kwargs)
        return ax

    def covariance_matrix(self, x, y, names=None, cov=None):
        """build a pyemu.Cov instance implied by Vario2d
        Args:
            x ([`float`]): x-coordinate locations
            y ([`float`]): y-coordinate locations
            names ([`str`]): names of locations. If None, cov must not be None
            cov (`pyemu.Cov`): an existing Cov instance.  Vario2d contribution is added to cov
            in place
        Returns:
            `pyemu.Cov`: the covariance matrix for `x`, `y` implied by `Vario2d`
        Note:
            either `names` or `cov` must not be None.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert x.shape[0] == y.shape[0]

        if names is not None:
            assert x.shape[0] == len(names)
            c = np.zeros((len(names), len(names)))
            np.fill_diagonal(c, self.contribution)
            cov = Cov(x=c, names=names)
        elif cov is not None:
            assert cov.shape[0] == x.shape[0]
            names = cov.row_names
            c = np.zeros((len(names), 1)) + self.contribution
            cont = Cov(x=c, names=names, isdiagonal=True)
            cov += cont

        else:
            raise Exception(
                "Vario2d.covariance_matrix() requires either" + "names or cov arg"
            )
        rc = self.rotation_coefs
        for i1, (n1, x1, y1) in enumerate(zip(names, x, y)):
            dx = x1 - x[i1 + 1 :]
            dy = y1 - y[i1 + 1 :]
            dxx, dyy = self._apply_rotation(dx, dy)
            h = np.sqrt(dxx * dxx + dyy * dyy)

            h[h < 0.0] = 0.0
            h = self._h_function(h)
            if np.any(np.isnan(h)):
                raise Exception("nans in h for i1 {0}".format(i1))
            cov.x[i1, i1 + 1 :] += h
        for i in range(len(names)):
            cov.x[i + 1 :, i] = cov.x[i, i + 1 :]
        return cov

    def _specsim_grid_contrib(self, grid):
        rot_grid = grid
        if self.bearing % 90.0 != 0:
            dx, dy = self._apply_rotation(grid[0, :, :], grid[1, :, :])
            rot_grid = np.array((dx, dy))
        h = ((rot_grid ** 2).sum(axis=0)) ** 0.5
        c = self._h_function(h)
        return c

    def _apply_rotation(self, dx, dy):
        """private method to rotate points
        according to Vario2d.bearing and Vario2d.anisotropy
        """
        if self.anisotropy == 1.0:
            return dx, dy
        rcoefs = self.rotation_coefs
        dxx = (dx * rcoefs[0]) + (dy * rcoefs[1])
        dyy = ((dx * rcoefs[2]) + (dy * rcoefs[3])) * self.anisotropy
        return dxx, dyy

    def covariance_points(self, x0, y0, xother, yother):
        """get the covariance between base point (x0,y0) and
        other points xother,yother implied by `Vario2d`
        Args:
            x0 (`float`): x-coordinate
            y0 (`float`): y-coordinate
            xother ([`float`]): x-coordinates of other points
            yother ([`float`]): y-coordinates of other points
        Returns:
            `numpy.ndarray`: a 1-D array of covariance between point x0,y0 and the
            points contained in xother, yother.  len(cov) = len(xother) =
            len(yother)
        """
        dxx = x0 - xother
        dyy = y0 - yother
        dxx, dyy = self._apply_rotation(dxx, dyy)
        h = np.sqrt(dxx * dxx + dyy * dyy)
        return self._h_function(h)

    def covariance(self, pt0, pt1):
        """get the covarince between two points implied by Vario2d
        Args:
            pt0 : ([`float`]): first point x and y
            pt1 : ([`float`]): second point x and y
        Returns:
            `float`: covariance between pt0 and pt1
        """

        x = np.array([pt0[0], pt1[0]])
        y = np.array([pt0[1], pt1[1]])
        names = ["n1", "n2"]
        return self.covariance_matrix(x, y, names=names).x[0, 1]

    def __str__(self):
        """get the str representation of Vario2d
        Returns:
            `str`: string rep
        """
        s = "name:{0},contribution:{1},a:{2},anisotropy:{3},bearing:{4}\n".format(
            self.name, self.contribution, self.a, self.anisotropy, self.bearing
        )
        return s

class ExpVario(Vario2d):
    """Exponential variogram derived type
    Args:
        contribution (float): sill of the variogram
        a (`float`): (practical) range of correlation
        anisotropy (`float`, optional): Anisotropy ratio. Default is 1.0
        bearing : (`float`, optional): angle in degrees East of North corresponding
            to anisotropy ellipse. Default is 0.0
        name (`str`, optinoal): name of the variogram.  Default is "var1"
    Example::
        v = pyemu.utils.geostats.ExpVario(a=1000,contribution=1.0)
    """

    def __init__(self, contribution, a, anisotropy=1.0, bearing=0.0, name="var1"):
        super(ExpVario, self).__init__(
            contribution, a, anisotropy=anisotropy, bearing=bearing, name=name
        )
        self.vartype = 2

    def _h_function(self, h):
        """private method exponential variogram "h" function"""
        return self.contribution * np.exp(-1.0 * h / self.a)

def variogram(x,y):
    return 0


def main():

    x,y = [],[]






