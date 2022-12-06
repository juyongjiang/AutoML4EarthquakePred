import os
import random
import numpy as np
from time import strptime, mktime, localtime, strftime


# string <-> stamp
def string2stamp(timeString):
    _timeTuple = strptime(timeString, "%Y%m%d")
    return int(mktime(_timeTuple))

def stamp2string(timeStamp):
    _timeTuple = localtime(float(timeStamp))
    return strftime("%Y%m%d", _timeTuple)
