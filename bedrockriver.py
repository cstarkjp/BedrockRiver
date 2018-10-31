try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    get_ipython().magic('aimport bedrockriver')
except:
    pass
try:
    get_ipython().magic('matplotlib inline')
except:
    pass    
try:
    get_ipython().magic("config InlineBackend.figure_format = 'retina'")
except:
    pass

import plot

from symbols import *
from utils import *

import sinuosity
import ensemble