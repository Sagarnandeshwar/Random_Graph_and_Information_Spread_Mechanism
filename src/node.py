import numpy as np

"""
Node Class
"""
class Node:
    def __init__(self, ID, group, position, init_influence, postThreshold):
        self.ID = ID
        self.group = group
        self.position = position
        self.follower = []
        self.following = []
        self.influenceFactor = init_influence
        self.view = {}
        self.post = {}
        self.postThreshold = postThreshold

    """
    add a message to the view
    """
    def addView(self, packet, budget):
        packet_id = packet.ID
        if packet_id not in self.view:
            self.view[packet_id] = {"data": packet, "count": 1, "budget": budget}
        else:
            self.view[packet_id]["count"] = self.view[packet_id]["count"] + 1
            self.view[packet_id]["budget"] = self.view[packet_id]["budget"] + budget

    """
        move message from view to post if the budget of the message is more than self.postThreshold
    """
    def updatePost(self, msgID):
        if msgID in self.view.keys():
            if self.view[msgID]["budget"] > self.postThreshold:
                self.post[msgID] = {"data": self.view[msgID]["data"], "budget": self.view[msgID]["budget"]}
