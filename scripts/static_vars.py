# -*- coding: utf-8 -*-
"""
Package static_vars
===================

A decorator for adding static variables to a function.
see https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
"""
__version__ = "0.2.0"


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# eof
