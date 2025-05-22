#!/usr/bin/python
# -*- coding: utf-8 -*-

## Classe time : conversion d'une date et jours juliens
## et d'un jour julien en une date, plus d'autres fonctions de base
## Théo Vaquer 2010/03 
## Charly REGNIER  2012/07 Création classe qui démarre de 1900

# standard package
import datetime 
import re

# local package
from . import syerrors as syerr

__all__= ('SyDate',)


# =====
# Class
# =====

class SyDate(object):
    """Time class for manipulating date in Mercator uses.

     For convenience, we can find fonctions in two form :
     - the first one is provided with the historic algorithm,
     these fonction's names start with an '_'
     - and the second form is more python compliant, these will be more faster and,
     related with the python implementation, more reliable.

     The class is writen for further python 3000 and stay compatible with
     python 2.5
     """
    _julianorigin = datetime.date(1950, 1, 1).toordinal()
   
    ## Init 
    def __init__(self, p_date):
        if SyDate.__check(p_date): 
           self._date = datetime.date(int(p_date[0:4]),
                             int(p_date[4:6]),
                             int(p_date[6:8]))
        else: 
            raise syerr.SytoolkitDateFormatError(p_date)
    #def __init__(self,p_date,p_dateorigin=1950):
    #    if SyDate.__check(p_date): 
    #       self._date = datetime.date(int(p_date[0:4]),
    #                         int(p_date[4:6]),
    #                         int(p_date[6:8]))
    #    else: 
    #        raise syerr.SytoolkitDateFormatError, p_date
    #    if SyDate.__check2(p_dateorigin):
    #       # The origin day for Mercator's julian calendar
    #       self.__julianorigin = datetime.date(p_dateorigin, 1, 1).toordinal()
    #       print 'test'
    #    else:
    #       raise syerr.SytoolkitDateFormatError, p_dateorigin

    def __str__(self):
        return self._date.strftime("%Y%m%d")


    def __repr__(self):
        return SyDate.__name__ + '("' + str(self) + '")'
   

    def __eq__(self, p):
        if isinstance(p, SyDate):
            return self._date == p._date
        else:
            raise TypeError("can compare only with SyDate object")

    def __lt__(self, p):
        if isinstance(p, SyDate):
            return self._date < p._date
        else:
            raise TypeError("can compare only with SyDate object")

    def __gt__(self, p):
        if isinstance(p, SyDate):
            return self._date > p._date
        else:
            raise TypeError("can compare only with SyDate object")

    def __le__(self, p):
        if isinstance(p, SyDate):
            return self._date <= p._date
        else:
            raise TypeError("can compare only with SyDate object")

    def __ge__(self, p):
        if isinstance(p, SyDate):
            return self._date >= p._date
        else:
            raise TypeError("can compare only with SyDate object")




    # ===============
    # Properties
    #================
    @property
    def year(self):
        return self._date.year

    @property
    def month(self):
        return self._date.month

    @property
    def day(self):
        return self._date.day

    # ===============
    # Instance method
    # ===============

    # move the date of n day(s) in past
    def setbackward(self, nbday):
        """This will set the date to n day in past.
        Exemple : 2010-03-17 setbackward(7) result 2010-03-10"""
        self._date -= datetime.timedelta(nbday)
    
    # move the date of n day(s) in future
    def setforward(self, nbday):
        """This will set the date to n day in future.
        Exemple : 2010-03-17 setforward(7) result 2010-03-24"""
        self._date += datetime.timedelta(nbday)
 
    # return a new date of n day(s) in past
    def gobackward(self, nbday):
        """return a new date, n day in past.
        Exemple : 2010-03-17 gobackward(7) result 2010-03-10"""
        result = self._date - datetime.timedelta(nbday)
        return SyDate(result.strftime("%Y%m%d"))

    # return a new date of n day(s) in future
    def goforward(self, nbday):
        """return a new date, n day in future.
        Exemple : 2010-03-17 goforward(7) result 2010-03-24"""
        result = self._date + datetime.timedelta(nbday)
        return SyDate(result.strftime("%Y%m%d"))
   
    # Faster way, same result ... in theory
    def tojulian(self):
        """Return the number of day passed since 1950-01-01"""
        return (self._date.toordinal() - self._julianorigin)

    def setfromjulian(self, p_julian):
        self._date = self._date.fromordinal(p_julian + self._julianorigin)
        return self


    # Test the period time name
    def isforecast(self, p):
        """Define if p is in Forecast period (d+1 to d+7), 
        the reference date is self._date.
        
        """
        SyDate.__checkType(p)
        return p._date > self._date
    
    def ishindcast(self, p):
        """Define if p is in hindcast period (d-13 to d-7)
        the reference date is self._date.

        """
        SyDate.__checkType(p)
        length = self._date - p._date
        return (p._date < self._date and length >= datetime.timedelta(7))

    def isnowcast(self, p):
        """Define if p is in nowcast period (d-6 to d)
        the reference date is self._date.
        
        """
        SyDate.__checkType(p)
        length = self._date - p._date
        return ( p._date <= self._date and 
                 (length < datetime.timedelta(7) 
                  and length >= datetime.timedelta(0)) )

    
    # creator
    #@classmethod
    #def fromjulian(cls, p_julian):
    #    """Create a SyDate in gregorian format from a julian date"""
    #    juliandate = datetime.date.fromordinal(p_julian + cls._julianorigin)
    #    return SyDate(juliandate.strftime("%Y%m%d"))
    @classmethod
    def fromjulian(cls, p_julian,p_date):
        __julianorigin=datetime.date(p_date, 1, 1).toordinal()
        """Create a SyDate in gregorian format from a julian date"""
        juliandate = datetime.date.fromordinal(p_julian + __julianorigin)
        return SyDate(juliandate.strftime("%Y%m%d"))

    # ================
    # Static method
    # ================
    @staticmethod
    def __check(date):
        """Check if date is a string and 
        if is matching YYYYMMDD format (20100317)
        
        """
        regexp = r'^([0-9]{4})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])'
        return isinstance(date, str) and re.match(regexp, date)

    @staticmethod
    def __check2(date):
        """Check if date is a string and 
        if is matching YYYYMMDD format (20100317)
        
        """
        regexp = r'^([0-9]{2})(0[1-9]|1[0-2])'
        return isinstance(date,int) and re.match(regexp, date)

    @staticmethod
    def __checkType(pSyDate):
        if not isinstance(pSyDate, SyDate):
            raise TypeError("bad argument type")

    # Retro-compatibility
    # ~~~~~~~~~~~~~~~~~~

    # Same, but use the old conversion method. May be more safe.
    @staticmethod
    def _fromjulian(p_julian):
        """Create a SyDate in gregorian format from a julian date"""
        tmpday =  p_julian + 712164
        centuries = (4 * tmpday - 1) / 146097
        tmpday = tmpday + (centuries - centuries / 4)
        year = (4 * tmpday - 1) / 1461
        tmpday = tmpday -((1461 * year) / 4)
        month = (10 * tmpday - 5) / 306
        day = tmpday - (306 * month + 5) / 10
        month = month + 2
        year = year+month / 12
        month = month % 12 + 1
        # Need to format the results int
        str_result = str(year) + str(month).zfill(2) + str(day).zfill(2)
        return SyDate(str_result)

    # Old algorithm, keeped for safety
    def _setfromjulian(self, p_julian):
        """Set the gregorian date, from a Mercator's julian day."""
        ## Based on current scripts
        tmpday =  p_julian + 712164
        centuries = (4 * tmpday - 1) / 146097
        tmpday = tmpday + (centuries - centuries / 4)
        year = (4 * tmpday - 1) / 1461
        tmpday = tmpday -((1461 * year) / 4)
        month = (10 * tmpday - 5) / 306
        day = tmpday - (306 * month + 5) / 10
        month = month + 2
        year = year+month / 12
        month = month % 12 + 1
        # update current object with the new date
        self._date = self._date.replace(year, month, day)
        return self
    
    def _tojulian(self):
        """Return the number of day passed since Mercator's julian day
        origin (ie: 1950-01-01).
        """
        ## Based on current scripts
        tmpmonth = (12 * self.year) + self.month - 3
        tmpyear = tmpmonth / 12
        val = ((734 * tmpmonth + 15) / 24 -
                2 * tmpyear + tmpyear/4 - tmpyear/100 +
                tmpyear/400 + self.day - 712164) 
        return val
#end SyDate 


# vim: set ts=4 sw=4 tw=79:


