# -*- coding: UTF-8 -*-

""" Exception classes for sytoolkit """

__all__ = (
        'SytoolkitException',
        'SytoolkitFatalError',
        'SytoolkitMTOConnectTimeOut',
        'SytoolkitMTOConnectEOF',
        'SytoolkitDateFormatError',
)

# ==========
# Exceptions
# ==========

class SytoolkitException(Exception):
    """ Base class for sytoolkit exceptions."""
    def __init__(self):
        pass


class SytoolkitFatalError(SytoolkitException):
    """A fatal error occured, usually this means a direct exit."""
    def __init__(self, err):
        self.err = err

    def __str__(self):
        return "Fatal error: %s" % self.err


class SytoolkitMTOConnectTimeOut(SytoolkitException):
    """Got a time our when connecting to meteo's host."""
    def __init__(self, host):
        self.host = host

    def __str__(self):
        return "Time out error: can't connect to %s" % self.host

class SytoolkitMTOConnectEOF(SytoolkitException):
    """Got a EOF during connection with meteo's host, this means 
    in most case that we use bad login and/orpassword."""
    def __init__(self, host):
        self.host = host

    def __str__(self):
        return ("EOF error: connection with %s has been remotly closed. "\
                "Check your permission and/or password to this host." % self.host)

class SytoolkitDateFormatError(SytoolkitException):
    """A wrong date format is used."""
    def __init__(self, date):
        self.date = date

    def __str__(self):
        return "Date format error: %s" % self.date



# vim: set ts=4 sw=4 tw=79:
