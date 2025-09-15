from enum import Enum


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from colorama import Fore, Back, Style

class Color(object):
    
    WARN = Fore.YELLOW
    ERROR = Fore.RED
    INFO = Fore.BLUE
    RESET = Style.RESET_ALL
    OK = Fore.GREEN
    CYAN = Fore.CYAN
    MAGENTA = Fore.MAGENTA
    BLACK = Fore.BLACK


def printH(header:str, message:str, type:str, end:str="\n"):
    
    def type2color(t):
        if t == "w":
            return Color.WARN
        elif t == "e":
            return Color.ERROR
        elif t=="i":
            return Color.INFO
        elif t == "o":
            return Color.OK
        elif t == "c":
            return Color.CYAN
        elif t=="m":
            return Color.MAGENTA
        elif t=="b":
            return Color.BLACK
        else:
            return ""
        
    print(f"{type2color(type)}{header}{Color.RESET} {message}", end=end)
    

