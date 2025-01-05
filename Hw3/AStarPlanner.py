import numpy as np
import heapq


class AStarPlanner(object):
    def __init__(self, bb, start, goal):
        self.bb = bb
        self.start = start
        self.goal = goal

        self.nodes = dict()

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # define all directions the agent can take - order doesn't matter here
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]

        self.epsilon = 20
        plan = self.a_star(self.start, self.goal)
        return np.array(plan)

    # compute heuristic based on the planning_env
    def compute_heuristic(self, state):
        '''
        Return the heuristic function for the A* algorithm.
        @param state The state (position) of the robot.
        '''
        return self.bb.compute_distance(state, self.goal)

    def a_star(self, start_loc, goal_loc):
        start_loc = tuple(start_loc)
        goal_loc = tuple(goal_loc)
        heap = [(0, start_loc)]
        distances = {}
        fathers = {}
        distances[start_loc] = 0
        visited = set()
        goal_reached = False
        while heap:
            _, vertex = heapq.heappop(heap)
            current_distance = distances[vertex]
            self.expanded_nodes.append(vertex)
            if vertex in visited:
                continue
            if np.array_equal(vertex, goal_loc):
                goal_reached = True
                break
            visited.add(vertex)
            for add_x, add_y in self.directions:
                x = vertex[0] + add_x
                y = vertex[1] + add_y
                if not self.bb.config_validity_checker((x, y)) or not self.bb.edge_validity_checker(vertex, (x, y)):
                    continue
                if add_x == 0 or add_y == 0:
                    g_score = current_distance + 1
                else:
                    g_score = current_distance + np.sqrt(2)
                if (x, y) not in distances.keys() or g_score < distances[(x, y)]:
                    distances[(x, y)] = g_score
                    fathers[(x, y)] = vertex
                    heapq.heappush(heap, (
                        g_score + self.epsilon * self.compute_heuristic(np.array([x, y])), (x, y)))
        if goal_reached:
            stack = [goal_loc]
            while stack[-1] != start_loc:
                stack.append(fathers[stack[-1]])
            path = []
            while stack:
                path.append(stack.pop())
            print(f'cost: {distances[goal_loc]}')
            print(f'expanded nodes: {len(self.expanded_nodes)}')
            return path
        return None, None

    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes
