import itertools

import networkx as nx
from node import Node
from edge import Edge
from message import MSG

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import json

import csv

"""
Class to generate the random Graph
"""
class Graph():
    def __init__(self, similarityProb, D_dimension, strict, activeStateProb, similarityFactor, distanceBoundry, distanceFactor, influenceFactor):
        self.G = nx.DiGraph()

        self.nodes = {}
        self.edges = {}

        self.similarityProb = similarityProb
        self.D_dimension = math.sqrt((D_dimension**2)+(D_dimension**2))
        self.strict = strict
        self.activeStateProb = activeStateProb
        self.similarityFactor = similarityFactor
        self.boundary = distanceBoundry
        self.distanceFactor = distanceFactor
        self.influenceFactor = influenceFactor

    """
    Add nodes to the graph during initial phase
    """
    def addInitialNode(self, node):
        id = node.ID
        self.G.add_node(id, data=node)
        self.nodes[id] = node

    """
    Add edges between the graph during initial phase; not considering nodes influence
    """
    def addInitialEdges(self):
        for node_i_id in self.nodes.keys():
            for node_j_id in self.nodes.keys():
                if node_i_id == node_j_id:
                    continue
                prob_edge_from_i_to_j = self.calculateEdgeProbability(self.nodes[node_i_id], self.nodes[node_j_id], True)
                edge_from_i_to_j = round(np.random.binomial(self.strict, prob_edge_from_i_to_j) / self.strict)
                if edge_from_i_to_j:
                    edgeData = Edge(self.nodes[node_i_id], self.nodes[node_j_id])
                    self.G.add_edge(node_i_id, node_j_id, data=edgeData)
                    self.edges[(node_i_id, node_j_id)] = edgeData
                    self.nodes[node_i_id].follower.append(self.nodes[node_j_id].ID)
                    self.nodes[node_j_id].following.append(self.nodes[node_i_id].ID)

    """
    Add nodes to the graph during expansion phase
    """
    def addNode(self, node, connect = False):
        id = node.ID
        self.G.add_node(id, data=node)
        self.nodes[id] = node
        if connect:
            self.addEdgesToNode(node)

    """
    add edge between given node and rest of the nodes
    """
    def addEdgesToNode(self, node):
        for nodes_id in self.nodes.keys():
            if node.ID == nodes_id:
                continue
            self.addEdge(node, self.nodes[nodes_id], True)
            self.addEdge(self.nodes[nodes_id], node, False)

    """
    get the edge probability and build and line between two given node
    """
    def addEdge(self, node_i, node_j, init_factor):
        prob = self.calculateEdgeProbability(node_i, node_j, init_factor)
        edge_from_i_to_j = round(np.random.binomial(self.strict, prob) / self.strict)
        if edge_from_i_to_j:
            edgeData = Edge(node_i.ID, node_j.ID)
            self.G.add_edge(node_i.ID, node_j.ID, data=edgeData)
            self.edges[(node_i.ID, node_i.ID)] = edgeData
            node_i.follower.append(node_j.ID)
            node_j.following.append(node_i.ID)

    """
    Calculated the edge probability between the nodes
    """
    def calculateEdgeProbability(self, node_i, node_j, init_factor):
        distanceProb = self.calculateDistance(node_i, node_j)
        similarityProb = self.calculateSimilarity(node_i, node_j)
        influence_i = self.calculateInfluence(node_i, init_factor)
        prob = np.mean([distanceProb, similarityProb, influence_i])

        """print("")
        print(node_i.ID)
        print(node_j.ID)
        print("Distance: " + str(distanceProb))
        print("Similarity: " + str(similarityProb))
        print("Influence" + str(influence_i))
        print(prob)"""

        return max(min(prob, 0.99), 0.01)

    """
    Calculated the distance factor between the nodes
    """
    def calculateDistance(self, node_i, node_j):
        pos1 = node_i.position
        pos2 = node_j.position
        distance = max(math.dist(pos1, pos2), 1)
        distance = 1 - (distance / self.D_dimension)
        distance = distance + ((distance - self.boundary)*self.distanceFactor)
        return distance

    """
    Calculated the similarity factor between the nodes
    """
    def calculateSimilarity(self, node_i, node_j):
        if node_i.group == node_j.group:
            return 0.5 + (self.similarityProb * self.similarityFactor)

        else:
            return 0.5 - (self.similarityProb * self.similarityFactor)

    """
        Calculated the influence factor of sender node
    """
    def calculateInfluence(self, node, init_factor):
        if init_factor:
            return node.influenceFactor
        return (len(node.follower)/len(self.nodes)) * (len(node.follower)/max(len(node.following), 1))

    """
    (seed) The start of the spread machenism
    """
    def seed(self, rootID, msgID):
        # Travel Path
        path = {"node": [], "edge": []}

        # BFS
        node_visited = {}
        for nodeID in self.nodes.keys():
            node_visited[nodeID] = False

        node_visited[rootID] = True

        que = []
        que.append(rootID)

        path["node"].append(rootID)

        while len(que) != 0:
            cur_node_ID = que.pop(0)
            cur_node = self.nodes[cur_node_ID]
            cur_node.updatePost(msgID)
            neighbours = cur_node.follower

            for neighbour_node_ID in neighbours:
                if not node_visited[neighbour_node_ID]:
                    active = np.random.binomial(1, self.activeStateProb)
                    if not active:
                        continue
                    node_visited[neighbour_node_ID] = True
                    if msgID in cur_node.post.keys():
                        weight = self.getEdgeWeight(cur_node_ID, neighbour_node_ID, False)
                        new_budget = cur_node.post[msgID]["budget"] - weight
                        self.nodes[neighbour_node_ID].addView(cur_node.post[msgID]["data"], new_budget)
                        path["node"].append(neighbour_node_ID)
                        path["edge"].append((cur_node_ID, neighbour_node_ID))
                    que.append(neighbour_node_ID)
        return path

    """
        Give information about nodes followers, following, view and posts
    """
    def detectGroup(self, nodeID):
        node = self.nodes[nodeID]

        infoTable = {"follower": {}, "following": {}, "view": {}, "post": {}}

        follower = node.follower
        following = node.following
        view = node.view
        post = node.post

        for a_node_id in follower:
            a_node = self.nodes[a_node_id]
            a_group = a_node.group
            if a_group not in infoTable["follower"].keys():
                infoTable["follower"][a_group] = 1
            else:
                infoTable["follower"][a_group] = infoTable["follower"][a_group] + 1

        for a_node_id in following:
            a_node = self.nodes[a_node_id]
            a_group = a_node.group
            if a_group not in infoTable["following"].keys():
                infoTable["following"][a_group] = 1
            else:
                infoTable["following"][a_group] = infoTable["following"][a_group] + 1

        for a_view_id in view:
            a_view = node.view[a_view_id]["data"]
            srcNode_id = a_view.srcID
            a_group = self.nodes[srcNode_id].group
            if a_group not in infoTable["view"].keys():
                infoTable["view"][a_group] = 1
            else:
                infoTable["view"][a_group] = infoTable["view"][a_group] + 1

        for a_view_id in view:
            a_view = node.view[a_view_id]["data"]
            srcNode_id = a_view.srcID
            a_group = self.nodes[srcNode_id].group
            if a_group not in infoTable["post"].keys():
                infoTable["post"][a_group] = 1
            else:
                infoTable["post"][a_group] = infoTable["post"][a_group] + 1

        return infoTable

    """
    visualize the graph
    """
    def visualize(self, fname = "model" , effect = False):
        if effect:
            position = {}
            for node_id in self.nodes.keys():
                position[node_id] = self.nodes[node_id].position

            colour_map = {}
            for node_id in self.nodes.keys():
                colour_map[node_id] = self.getColour(self.nodes[node_id].group)

            colour_map_list = []
            for node_keys in colour_map.keys():
                colour_map_list.append(colour_map[node_keys])

            w_size = {}
            weight = []
            for node_id in self.nodes.keys():
                w = (len(self.nodes[node_id].follower)/len(self.nodes.keys())) * (len(self.nodes[node_id].follower)/max(len(self.nodes[node_id].following), 1))
                weight.append(w*500)

            # print(w_size)
            nx.draw(self.G, position, node_color=colour_map_list, node_size=weight, with_labels = True)
            plt.savefig(fname)
            plt.show()
        else:
            nx.draw(self.G)
            plt.show()

    @staticmethod
    # Define the Colour Scheme
    def getColour(group):
        if group == "A":
            return "red"
        elif group == "B":
            return "blue"
        elif group == "C":
            return "green"
        return "black"

    """
    Calculated the weight of the edge
    """
    def getEdgeWeight(self, node_i_id, node_j_id, initFactor):
        if initFactor:
            influence = 1
        else:
            influence_i = self.calculateInfluence(self.nodes[node_i_id], False)
            influence_j = self.calculateInfluence(self.nodes[node_j_id], False)
            influence = influence_j / influence_i
        w = 0
        if self.nodes[node_i_id].group == self.nodes[node_j_id].group:
            w = 50 * influence
        else:
            w = 60 * influence
        return w

    """
        calculate the bottleNeck in the information flow
    """
    def bottleNeck(self, path):
        bn = []
        for a_node_id in path["node"]:
            a_node = self.nodes[a_node_id]
            if len(a_node.view.keys()) != 0:
                if len(a_node.post.keys()) == 0:
                    bn.append(a_node_id)
        return bn

    """
       visualize information flow 
    """
    def visualize_flow(self, path, fname):
        travel_path = nx.DiGraph()

        position = {}
        colour_map = {}

        for a_node in path["node"]:
            position[a_node] = self.nodes[a_node].position
            colour_map[a_node] = self.getColour(self.nodes[a_node].group)

        colour_map_list = []
        for node_keys in colour_map.keys():
            colour_map_list.append(colour_map[node_keys])

        weight = []
        for a_node in path["node"]:
            w = len(self.nodes[a_node].follower) / len(self.nodes) * len(self.nodes[a_node].follower) / max(len(self.nodes[a_node].following), 1)
            weight.append(w*500)
        travel_path.add_nodes_from(path["node"])
        travel_path.add_edges_from(path["edge"])
        nx.draw(travel_path, position, node_color=colour_map_list, node_size=weight, with_labels = True)
        plt.savefig(fname)
        plt.show()

    """
        calculate the influence of first n nodes
    """
    def powerLaw(self, n):
        infoTable = {}
        for nodeID in range(n):
            w = len(self.nodes[nodeID].follower) / len(self.nodes) * len(self.nodes[nodeID].follower) / max(len(self.nodes[nodeID].following), 1)
            infoTable[nodeID] = w
        return infoTable

    """
        save the information about influence of nodes
    """
    def powerLawSave(self, lis):
        power_dict = {}
        for ind in range(len(lis)):
            power_dict[ind] = lis[ind]

        with open('./../analysis/powerLaw.json', 'w') as f:
            json.dump(power_dict, f)

    """
        save the information about network density
    """
    def densitySave(self, densityList):
        density_dict = {"density": [], "nodes": [], "edges": [], "degree":[]}
        for ind in range(len(densityList)):
            density_dict["density"].append(densityList[ind][0])
            density_dict["nodes"].append(densityList[ind][1])
            density_dict["edges"].append(densityList[ind][2])
            density_dict["degree"].append(densityList[ind][3])

        with open('./../analysis/density.json', 'w') as f:
            json.dump(density_dict, f)

    """
        Calculate Degree Centrality
    """
    def degreeCentrality(self):
        data = nx.degree_centrality(self.G)
        with open('./../analysis/degreeCentrality.json', 'w') as f:
            json.dump(data, f)

    """
            Calculate Closeness Centrality
    """
    def closenessCentrality(self):
        data = nx.closeness_centrality(self.G)
        with open('./../analysis/closenessCentrality.json', 'w') as f:
            json.dump(data, f)

    """
        Calculate Eigenvector Centrality
    """
    def eigenvectorCentrality(self):
        data = nx.eigenvector_centrality(self.G)
        with open('./../analysis/eigenvectorCentrality.json', 'w') as f:
            json.dump(data, f)

    """
        Calculate betweeness Centrality
    """
    def betweennessCentrality(self):
        data = nx.betweenness_centrality(self.G)
        with open('./../analysis/betweennessCentrality.json', 'w') as f:
            json.dump(data, f)

    """
        Calculate no. of triangles for each node in graph  
    """
    def triangles(self):
        data = nx.triangles(self.G.to_undirected())
        with open('./../analysis/triangles.json', 'w') as f:
            json.dump(data, f)

    """
    calculate clustering coefficient for each node of the graph
    """
    def clustering(self):
        data = nx.clustering(self.G)
        with open('./../analysis/clustering.json', 'w') as f:
            json.dump(data, f)

    """
        calculate average clustering coefficient of the graph
    """
    def avgClustering(self):
        data = nx.average_clustering(self.G)
        return data

    """
    Calculated the diameter of the graph
    """
    def diameter(self):
        return nx.diameter(self.G)

    """
        return the least influential node of the graph
    """
    def getMin(self):
        minNode = 0
        minValue = 10000
        for nodeID in self.nodes.keys():
            w = len(self.nodes[nodeID].follower) / len(self.nodes) * len(self.nodes[nodeID].follower) / max(len(self.nodes[nodeID].following), 1)
            if w < minValue:
                minNode = nodeID
                minValue = w
        return minNode

    """
        return the most influential node of the graph
    """
    def getMax(self):
        maxNode = 0
        maxValue = -1
        for nodeID in self.nodes.keys():
            w = len(self.nodes[nodeID].follower) / len(self.nodes) * len(self.nodes[nodeID].follower) / max(len(self.nodes[nodeID].following), 1)
            if w > maxValue:
                maxNode = nodeID
                maxValue = w

        return maxNode

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


# Parameters
groupList = ["A", "B", "C"]
groupListProb = [0.33, 0.33, 0.33]

D_dimension = 1000

strictness = 5

similarity_score = 0.3
similarityFactor = 1

distanceBoundary = 0.85
distanceFactor = 1.2

influenceFactor = 1
activeStateProb = 1

if __name__ == "__main__":
    list1 = generate_init_random_nodes(0, 10, groupList, groupListProb, D_dimension, 0)
    list2 = generate_random_nodes(10, 20, groupList, groupListProb, D_dimension, 0)
    list3 = generate_random_nodes(20, 30, groupList, groupListProb, D_dimension, 0)
    list4 = generate_random_nodes(30, 40, groupList, groupListProb, D_dimension, 0)
    list5 = generate_random_nodes(40, 50, groupList, groupListProb, D_dimension, 0)
    list6 = generate_random_nodes(50, 60, groupList, groupListProb, D_dimension, 0)
    #list7 = generate_random_nodes(60, 70, groupList, groupListProb, D_dimension, 0)
    #list8 = generate_random_nodes(70, 80, groupList, groupListProb, D_dimension, 0)
    #list9 = generate_random_nodes(80, 160, groupList, groupListProb, D_dimension, 0)

    G1 = Graph(similarity_score, D_dimension, strictness, activeStateProb, similarityFactor, distanceBoundary, distanceFactor, influenceFactor)

    for node in list1:
        G1.addInitialNode(node)
    G1.addInitialEdges()

    p_n = 1
    p_list = []

    infoTable = {}
    for nodeID in range(p_n):
        w = 0.5
        infoTable[nodeID] = w
    p_list.append(infoTable)

    densityList = []

    p_list.append(G1.powerLaw(p_n))
    avg_degree = np.mean([d for _, d in G1.G.degree()])
    densityList.append((nx.density(G1.G), G1.G.number_of_nodes(), G1.G.number_of_edges(), avg_degree))

    #G1.visualize("1", True)

    for node in list2:
        G1.addNode(node, True)

    p_list.append(G1.powerLaw(p_n))
    avg_degree = np.mean([d for _, d in G1.G.degree()])
    densityList.append((nx.density(G1.G), G1.G.number_of_nodes(), G1.G.number_of_edges(), avg_degree))

    #G1.visualize("2", True)

    for node in list3:
        G1.addNode(node, True)

    p_list.append(G1.powerLaw(p_n))
    avg_degree = np.mean([d for _, d in G1.G.degree()])
    densityList.append((nx.density(G1.G), G1.G.number_of_nodes(), G1.G.number_of_edges(), avg_degree))

    #G1.visualize("3", True)

    for node in list4:
        G1.addNode(node, True)

    p_list.append(G1.powerLaw(p_n))
    avg_degree = np.mean([d for _, d in G1.G.degree()])
    densityList.append((nx.density(G1.G), G1.G.number_of_nodes(), G1.G.number_of_edges(), avg_degree))

    #G1.visualize("4", True)

    for node in list5:
        G1.addNode(node, True)

    #G1.visualize("5", True)

    p_list.append(G1.powerLaw(p_n))
    avg_degree = np.mean([d for _, d in G1.G.degree()])
    densityList.append((nx.density(G1.G), G1.G.number_of_nodes(), G1.G.number_of_edges(), avg_degree))


    G1.powerLawSave(p_list)
    G1.densitySave(densityList)

    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G1.G))
    diameter = max(nx.eccentricity(G1.G, sp=shortest_path_lengths).values())
    print(diameter)
    average_path_lengths = [np.mean(list(spl.values())) for spl in shortest_path_lengths.values()]
    print(np.mean(average_path_lengths))


    G1.degreeCentrality()
    G1.closenessCentrality()
    G1.betweennessCentrality()
    G1.eigenvectorCentrality()
    G1.clustering()
    print(G1.avgClustering())
    G1.triangles()

    minNode = G1.getMin()
    maxNode = G1.getMax()
    print(minNode)
    print(maxNode)

    msg1 = MSG(1, minNode, "Climate Change is not real", 40)
    root1 = G1.nodes[minNode]
    root1.addView(msg1, msg1.initBudget)
    path1 = G1.seed(minNode, 1)
    G1.visualize_flow(path1, "smallroot")
    node1 = G1.bottleNeck(path1)
    print(len(node1))

    count = 0
    for nodeId in G1.nodes.keys():
        if 1 not in G1.nodes[nodeId].view.keys():
            count = count + 1
    print(count)

    msg2 = MSG(2, maxNode, "Climate Change is real", 40)
    root2 = G1.nodes[maxNode]
    root2.addView(msg2, msg2.initBudget)
    path2 = G1.seed(maxNode, 2)
    G1.visualize_flow(path2, "big")
    node2 = G1.bottleNeck(path2)
    print(len(node2))

    count = 0
    for nodeId in G1.nodes.keys():
        if 2 not in G1.nodes[nodeId].view.keys():
            count = count + 1
    print(count)

    print(G1.detectGroup(45))

    quit()








