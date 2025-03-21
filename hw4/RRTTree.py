import operator
import numpy


class RRTTree(object):

    def __init__(self, bb):
        self.bb = bb
        self.vertices = {}
        self.edges = dict()
        self.goal_exist = False

    def GetRootID(self):
        '''
        Returns the ID of the root in the tree.
        '''
        return 0

    def GetNearestVertex(self, config):
        '''
        Returns the nearest state ID in the tree.
        @param config Sampled configuration.
        '''
        dists = [self.bb.compute_distance(config, self.vertices[v].state) for v in self.vertices]
        # for v in self.vertices:
        #     v_node = self.vertices[v]
        #     dists.append(self.bb.edge_cost(config, v_node.state))

        vid, vdist = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid]

    def GetKNN(self, config, k):
        '''
        Return k-nearest neighbors
        @param config Sampled configuration.
        @param k Number of nearest neighbors to retrieve.
        '''
        # dists = []
        dists = [self.bb.compute_distance(config, self.vertices[vid].state) for vid in self.vertices]
        # for id in self.vertices:
        #     dists.append(self.bb.edge_cost(config, self.vertices[id].state))

        dists = numpy.array(dists)
        knnIDs = numpy.argpartition(dists, k)[:k]
        # knnDists = [dists[i] for i in knnIDs]

        return knnIDs, [self.vertices[vid].state for vid in knnIDs]

    def AddVertex(self, config):
        '''
        Add a state to the tree.
        @param real_conf: the real coords of the point in the hypertorus
        @param config Configuration to add to the tree.
        '''
        vid = len(self.vertices)
        # self.vertices.append(config)
        # self.vertices[vid] = RRTVertex(state=config)#, real_state=real_conf)
        self.vertices[vid] = RRTVertex(state=config)  # , real_state=real_conf)
        return vid

    def AddEdge(self, sid, eid, edge_cost):
        '''
        Adds an edge in the tree.
        @param sid start state ID
        @param eid end state ID
        '''
        self.edges[eid] = sid
        self.vertices[eid].set_cost(cost=self.vertices[sid].cost + edge_cost)
        self.vertices[sid].children.add(self.vertices[eid])

    def get_idx_for_config(self, config):
        '''
        Search for the vertex with the given config and return the index if exists
        @param config Configuration to check if exists.
        '''
        valid_idxs = [v_idx for v_idx, v in self.vertices.items() if (v.state == config).all()]
        if len(valid_idxs) > 0:
            return valid_idxs[0]
        return None

    def is_goal_exists(self, config):
        '''
        Check if goal exists.
        @param config Configuration to check if exists.
        '''
        goal_idx = self.get_idx_for_config(config=config)
        if goal_idx is not None:
            return True
        return False


class RRTVertex(object):

    def __init__(self, state, cost=0, inspected_points=None):
        self.state = state
        # self.real_state = real_state
        self.cost = cost
        self.inspected_points = inspected_points
        self.children = set()

    def set_cost(self, cost):
        '''
        Set the cost of the vertex.
        '''
        self.cost = cost
