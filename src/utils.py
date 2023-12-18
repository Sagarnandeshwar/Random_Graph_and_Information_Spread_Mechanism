import random
from node import Node
import numpy as np

"""
    generate nodes from id start to end for initial phase
"""
def generate_init_random_nodes(start, end, groupList, groupListProb, dim, postThreshold):
    nodeList = []
    for num in range(start, end):
        x = random.randint(0, dim-1)
        y = random.randint(0, dim-1)
        g = random.choices(groupList, groupListProb, k=1)[0]
        init_influence = 0.5
        node = Node(num, g, [x, y], init_influence, postThreshold)
        nodeList.append(node)
    return nodeList

"""
    generate nodes from id start to end for expansion phase
"""
def generate_random_nodes(start, end, groupList, groupListProb, dim, postThreshold):
    nodeList = []
    for num in range(start, end):
        x = random.randint(0, dim-1)
        y = random.randint(0, dim-1)
        g = random.choices(groupList, groupListProb, k=1)[0]
        init_influence = min(max(0.5+np.random.normal(0.0, 0.01, size=1)[0], 0.1), 0.9)
        node = Node(num, g, [x, y], init_influence, postThreshold)
        nodeList.append(node)
    return nodeList