import numpy as np
import random

"""
Edge Class
"""
class Edge:
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest
        # self.weight = random.randint(0, 100)
        # self.curLoad = 0
