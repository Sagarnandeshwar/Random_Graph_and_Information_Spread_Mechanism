import random

import networkx as nx
import matplotlib.pyplot as plt
import scipy
import numpy as np
import json
import pandas
from node import Node
from message import MSG

"""
Class to generate graph for twitter data
"""
class twitter():
    def __init__(self, filename, flow):
        self.G = nx.DiGraph()
        self.filename = filename
        self.dataset = self.processData(filename)

        self.nodes = {}
        self.edges = {}
        self.buildGraph(flow)

        # self.G = nx.read_edgelist(filename, create_using = nx.DiGraph(), nodetype=int)

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
            print(neighbours)
            for neighbour_node_ID in neighbours:
                if not node_visited[neighbour_node_ID]:
                    node_visited[neighbour_node_ID] = True
                    if msgID in cur_node.post.keys():
                        weight = self.getEdgeWeight(cur_node_ID, neighbour_node_ID)
                        new_budget = cur_node.post[msgID]["budget"] - weight
                        self.nodes[neighbour_node_ID].addView(cur_node.post[msgID]["data"], new_budget)
                        path["node"].append(neighbour_node_ID)
                        path["edge"].append((cur_node_ID, neighbour_node_ID))
                    que.append(neighbour_node_ID)
        return path

    """
    Calculated the weight of the edge
    """
    def getEdgeWeight(self, cur_node_ID, neighbour_node_ID):
        return random.randint(40,60)

    """
    Process the Twitter data and make edges for the graph
    """
    def processData(self, fileName):
        df = pandas.read_csv(fileName, header=None)
        db = []
        for index, row in df.iterrows():
            db.append((int(row[0]), int(row[1])))
        return db

    """
        Build the graph with class dataset
    """
    def buildGraph(self, flow):
        if not flow:
            self.G.add_edges_from(self.dataset)
        else:
            for edge in self.dataset:
                self.G.add_edge(edge[0], edge[1])
                node1 = Node(edge[0], "A", [0, 0], 0.5, 0)
                node2 = Node(edge[1], "A", [0, 0], 0.5, 0)
                if edge[0] not in self.nodes.keys():
                    self.nodes[node1.ID] = node1
                if edge[1] not in self.nodes.keys():
                    self.nodes[node2.ID] = node2
                self.nodes[node1.ID].follower.append(node2.ID)
                self.nodes[node2.ID].following.append(node1.ID)

    """
        visualize twitter data
    """
    def visualize(self, fname):
        nx.draw(self.G)
        plt.savefig(fname)
        plt.show()

    """
        visualize information flow in twitter data
    """
    def visualize_flow(self, path, fname):
        travel_path = nx.DiGraph()
        travel_path.add_nodes_from(path["node"])
        travel_path.add_edges_from(path["edge"])
        nx.draw(travel_path)
        plt.savefig(fname)
        plt.show()

    """
        Calculate Degree Centrality
    """
    def degreeCentrality(self):
        data = nx.degree_centrality(self.G)
        with open('./../analysis/twitterdegreeCentrality.json', 'w') as f:
            json.dump(data, f)

    """
                Calculate Closeness Centrality
    """
    def closenessCentrality(self):
        data = nx.closeness_centrality(self.G)
        with open('./../analysis/twitterclosenessCentrality.json', 'w') as f:
            json.dump(data, f)

    """
            Calculate Eigenvector Centrality
    """
    def eigenvectorCentrality(self):
        data = nx.eigenvector_centrality(self.G)
        with open('./../analysis/twittereigenvectorCentrality.json', 'w') as f:
            json.dump(data, f)

    """
        Calculate betweeness Centrality
    """
    def betweennessCentrality(self):
        data = nx.betweenness_centrality(self.G)
        with open('./../analysis/twitterbetweennessCentrality.json', 'w') as f:
            json.dump(data, f)

    """
        Calculate no. of triangles for each node in graph  
    """
    def triangles(self):
        data = nx.triangles(self.G.to_undirected())
        with open('./../analysis/twittertriangles.json', 'w') as f:
            json.dump(data, f)

    """
        calculate clustering coefficient for each node of the graph
    """
    def clustering(self):
        data = nx.clustering(self.G)
        with open('./../analysis/twitterclustering.json', 'w') as f:
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

if __name__ == "__main__":
    t = twitter('Twitter_Tweet.csv', True)
    print("START")
    root = 16
    msg1 = MSG(1, root, "Climate Change is not real", 1000)
    root1 = t.nodes[root]
    root1.addView(msg1, msg1.initBudget)
    path1 = t.seed(root, 1)
    print(path1)
    t.visualize_flow(path1, "twitter_flow")

    quit()
    t = twitter('Twitter_Politics.csv', False)
    print(t.G)
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(t.G))
    diameter = max(nx.eccentricity(t.G, sp=shortest_path_lengths).values())
    print(diameter)
    average_path_lengths = [np.mean(list(spl.values())) for spl in shortest_path_lengths.values()]
    print(np.mean(average_path_lengths))

    t.degreeCentrality()
    t.closenessCentrality()
    t.betweennessCentrality()
    t.eigenvectorCentrality()

    t.clustering()
    print(t.avgClustering())
    t.triangles()

    # t.visualize("twitter_map")


