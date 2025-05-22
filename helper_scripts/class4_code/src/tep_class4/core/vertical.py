"""
Vertical interpolators
======================

Vertical interpolation is achieved through use of :py:mod:`scipy.interpolate`
module. The module is a wrapper of the Fortran library FITPACK. However, it
does not preserve or indeed handle masked arrays elegantly.

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from window import Window, EmptyWindow
import sys


class Vertical2DInterpolator(object):
    """Vertical section interpolator

    Interpolates model grid section to observed section depths.
    Both sections should be 2D arrays whose first dimension represents
    the number of water columns. The second dimension of each section
    should be the depth dimension.

    .. note:: S-level and Z-level models are processed by this interpolator

    :param depths: array shaped ([N,] Z) where N is the number of profiles
                   and Z represents the model levels
    :param field: array shaped (N, Z)

    :returns: interpolator function
    """

    def __init__(self, depths, field):
        self.depths = depths
        self.field = field

    def __call__(self, observed_depths):
        """Apply vertical interpolation to estimate model values at observed
        depths

        :param observed_depths: array shaped (N, M)
        :returns: array shaped (N, M) where N is the number of profiles
                  and M is the number of observed levels
        """
        result = []
        for iprofile, observed_levels in enumerate(observed_depths):
            model_values = self.field[iprofile]
            if self.depths.ndim == 1:
                model_levels = self.depths
            else:
                model_levels = self.depths[iprofile]
            interpolator = Vertical1DInterpolator(model_levels, model_values)
            result.append(interpolator(observed_levels))
        return np.ma.MaskedArray(result)


class Vertical1DInterpolator(object):
    """Vertical water column interpolator

    Interpolates a single model water column to an observed profile.
    Masked data points are excluded from the interpolation but included
    in the result.

    :param depths: 1D array
    :param field: 1D array
    """

    def __init__(self, depths, field):
        assert len(depths) > 0, "must specify at least 1 level"
        depths, field = self.match(depths, field)
        self._interpolator = self.select_interpolator(depths, field)
        self.window = self.select_window(depths)

    @staticmethod
    def match(depths, field):
        """combine masks from both depths and field"""
        depths, field = np.ma.masked_invalid(np.ma.asarray(depths)), np.ma.masked_invalid(np.ma.asarray(field))
        common = depths.mask | field.mask
        return (np.ma.masked_array(depths, common), np.ma.masked_array(field, common))

    @staticmethod
    def select_window(depths):
        """select appropriate window function

        :returns: either :class:`obsoper.window.EmptyWindow`
                  or :class:`obsoper.window.Window`
        """
        if depths.mask.all():
            return EmptyWindow()
        return Window(depths.min(), depths.max())

    def select_interpolator(self, depths, field):
        """Select appropriate interpolation function

        ========= ===========================
        Knots     Interpolator
        ========= ===========================
        0 or 1    Masked data always returned
        2 or 3    Linear spline interpolator
        4 or more Cubic spline interpolator
        ========= ===========================

        Spline interpolators are created using non-masked data points

        :returns: interpolator chosen from above table
        """
        ## Todo : add log
        # print ('Nombre de noeuds {}'.format(self.knots(depths)))
        if self.knots(depths) < 2:
            # Masked data interpolator
            return lambda x: np.ma.masked_all(len(x))
        if self.knots(depths) < 4:
            # Linear spline interpolator
            # print "Linear spline interpolator"
            return InterpolatedUnivariateSpline(depths.compressed(), field.compressed(), k=1)
        # Cubic spline interpolator
        # print "Cubic spline interpolator"
        return InterpolatedUnivariateSpline(depths.compressed(), field.compressed())

    @staticmethod
    def knots(depths):
        """count number of knots"""
        return np.ma.count(depths)

    def __call__(self, observed_depths):
        """interpolate field to observed depths"""
        observed_depths = self.screen_depths(observed_depths)
        if observed_depths.mask.any():
            return self.masked_interpolate(observed_depths)
        else:
            return self._interpolator(observed_depths)

    def screen_depths(self, observed_depths):
        """mask depths outside model water column"""
        observed_depths = np.ma.asarray(observed_depths)
        mask = self.window.outside(observed_depths)
        return np.ma.masked_array(observed_depths, observed_depths.mask | mask)

    def masked_interpolate(self, observed_depths):
        """performs interpolation on valid data while preserving mask"""
        result = np.ma.masked_all(observed_depths.shape)
        points, values = (~observed_depths.mask, observed_depths.compressed())
        result[points] = self._interpolator(values)
        return result


class Section(Vertical2DInterpolator):
    """Vertical section interpolator

    Interpolates model grid section to observed section depths.
    Both sections should be 2D arrays whose first dimension represents
    the number of water columns. The second dimension of each section
    should be the depth dimension.

    .. note:: S-level and Z-level models are processed by this interpolator

    :param values: array shaped (N, Z) containing model values
    :param depths: array shaped ([N,] Z) where N is the number of profiles
                   and Z represents the model levels
    """

    def __init__(self, values, depths):
        assert np.shape(values)[-1] == np.shape(depths)[-1], "values and depths must have same levels"
        # Note: changed interface from (depths, values) to (values, depths)
        super(Section, self).__init__(depths, values)

    def __repr__(self):
        return "{}(values={},\ndepths={})".format(self.__class__.__name__, repr(self.values), repr(self.depths))

    @property
    def values(self):
        return self.field

    def interpolate(self, observed_depths):
        """Apply vertical interpolation to estimate model values at observed
        depths

        :param observed_depths: array shaped (N, M)
        :returns: array shaped (N, M) where N is the number of profiles
                  and M is the number of observed levels
        """
        if isinstance(observed_depths, list):
            observed_depths = np.asarray(observed_depths)
        message = ("observed depths has incorrect first dimension: " "{} != {}").format(
            observed_depths.shape[0], self.field.shape[0]
        )
        assert observed_depths.shape[0] == self.field.shape[0], message
        return self(observed_depths)


class Column(Vertical1DInterpolator):
    """Vertical column interpolator

    Interpolates a single model water column to an observed profile.
    Masked data points are excluded from the interpolation but included
    in the result.

    :param field: array shaped (N,) representing the water column values
    :param depths: array shaped (N,) representing the water column depths
    """

    def __init__(self, values, depths):
        # Note: changed interface from (depths, values) to (values, depths)
        super(Column, self).__init__(depths, values)

    def interpolate(self, observed_depths):
        """Apply vertical interpolation to estimate model values at observed
        depths

        :param observed_depths: array shaped (M,)
        :returns: array shaped (M,) where M is the number of observed levels
        """
        return self(observed_depths)
