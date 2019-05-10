#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          GPU.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190509
License:            MIT

Functionality not yet defined.
Should contain:
- GPU helper functions
- GPU fields?
"""
from inspect import signature
from org_namelist import wp_3D


def cuda_kernel_decorator(function):
    n_input_args = len(signature(function).parameters)
    decorator = 'void('
    for i in range(n_input_args):
        decorator += wp_3D+','
    decorator = decorator[:-1]
    decorator += ')'
    return(decorator)





class GPU:

    def __init__(self):
        pass
