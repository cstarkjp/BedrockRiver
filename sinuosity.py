import numpy as np                  
from scipy.interpolate import interp1d, interp2d
import sympy as sy
import scipy.integrate as si
from sympy.utilities.lambdify import lambdastr
from sympy.parsing.sympy_parser import parse_expr
from scipy.optimize import newton, curve_fit
import warnings

from symbols import *
from utils import *
from channel import channel_model

class bend_model(channel_model):
    pass