"""
Window filters
==============

Selects data values included or excluded by specific criteria.

"""
import datetime as dt
import numpy as np


class Window(object):
    """Window filter"""
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def inside(self, values):
        """determines if values are inside window"""
        return (values >= self.start) & (values < self.end)

    def outside(self, values):
        """determines if values are outside window"""
        return ~self.inside(values)


class TimeWindow(Window):
    """Time window"""
    @classmethod
    def fromtext(cls, start, end, fmt="%Y%m%d"):
        """construct from strings of a particular format"""
        return cls(dt.datetime.strptime(start, fmt),
                   dt.datetime.strptime(end, fmt))

    @classmethod
    def frominterval(cls, start, interval):
        """construct from start date and interval"""
        return cls(start, start + interval)


class EmptyWindow(object):
    """Empty window"""
    @staticmethod
    def inside(values):
        """always returns true"""
        return np.zeros(values.shape, dtype=np.bool)

    def outside(self, values):
        """always returns false"""
        return ~self.inside(values)
