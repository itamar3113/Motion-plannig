import argparse
import os
from typing import List, Tuple
import numpy as np
from Plotter import Plotter
from shapely.geometry.polygon import Polygon, LineString


def sort_vertices_counterclockwise(polygon):
    # center_x, center_y = polygon.mean(0)
    # x, y = polygon.T
    # angles = np.arctan2(y, x)
    # indices = np.argsort(angles)
    # return polygon[indices]
    return polygon


def cross_product(p1, p2):
    return p1[0] * p2[1] - p1[1] * p2[0]


def get_minkowsky_sum(original_shape: Polygon, r: float) -> Polygon:
    """
    Get the polygon representing the Minkowsky sum
    :param original_shape: The original obstacle
    :param r: The radius of the rhombus
    :return: The polygon composed from the Minkowsky sums
    """
    vertices = original_shape.exterior.coords[:-1]
    vertices = sort_vertices_counterclockwise(vertices)
    robot_in_origin = [(0, -r), (r, 0), (0, r), (-r, 0)]
    i, j = 0, 0
    res = []
    len_poly = len(vertices)
    len_robot = len(robot_in_origin)
    p1, next_poly = vertices[i]
    p2, next_robot = robot_in_origin[j]
    while i < len_poly or j < len_robot:
        p1 = vertices[i % len_poly]
        p2 = robot_in_origin[j % len_robot]
        res.append((p1[0] + p2[0], p1[1] + p2[1]))
        cross = cross_product((vertices[(i + 1) % len_poly][0] - vertices[i % len_poly][0],
                               vertices[(i + 1) % len_poly][1] - vertices[i % len_poly][1]),
                              (robot_in_origin[(j + 1) % len_robot][0] - robot_in_origin[j % len_robot][0],
                               robot_in_origin[(j + 1) % len_robot][1] - robot_in_origin[j % len_robot][1]))
        if cross >= 0:
            i += 1
        if cross <= 0:
            j += 1
    return Polygon(res)


# TODO
def get_visibility_graph(obstacles: List[Polygon], source=None, dest=None) -> List[LineString]:
    """
    Get The visibility graph of a given map
    :param obstacles: A list of the obstacles in the map
    :param source: The starting position of the robot. None for part 1.
    :param dest: The destination of the query. None for part 1.
    :return: A list of LineStrings holding the edges of the visibility graph
    """
    pass


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)


def get_points_and_dist(line):
    source, dist = line.split(' ')
    dist = float(dist)
    source = tuple(map(float, source.split(',')))
    return source, dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Robot",
                        help="A file that holds the starting position of the robot, and the distance from the center of the robot to any of its vertices")
    parser.add_argument("Obstacles", help="A file that contains the obstacles in the map")
    parser.add_argument("Query", help="A file that contains the ending position for the robot.")
    args = parser.parse_args()
    obstacles = args.Obstacles
    robot = args.Robot
    query = args.Query
    is_valid_file(parser, obstacles)
    is_valid_file(parser, robot)
    is_valid_file(parser, query)
    workspace_obstacles = []
    with open(obstacles, 'r') as f:
        for line in f.readlines():
            ob_vertices = line.split(' ')
            if ',' not in ob_vertices:
                ob_vertices = ob_vertices[:-1]
            points = [tuple(map(float, t.split(','))) for t in ob_vertices]
            workspace_obstacles.append(Polygon(points))
    with open(robot, 'r') as f:
        source, dist = get_points_and_dist(f.readline())

    # step 1:
    c_space_obstacles = [get_minkowsky_sum(p, dist) for p in workspace_obstacles]
    plotter1 = Plotter()

    plotter1.add_obstacles(workspace_obstacles)
    plotter1.add_c_space_obstacles(c_space_obstacles)
    plotter1.add_robot(source, dist)

    plotter1.show_graph()

    # step 2:

    # lines = get_visibility_graph(c_space_obstacles)
    # plotter2 = Plotter()
    #
    # plotter2.add_obstacles(workspace_obstacles)
    # plotter2.add_c_space_obstacles(c_space_obstacles)
    # plotter2.add_visibility_graph(lines)
    # plotter2.add_robot(source, dist)
    #
    # plotter2.show_graph()
    #
    # # step 3:
    # with open(query, 'r') as f:
    #     dest = tuple(map(float, f.readline().split(',')))
    #
    # lines = get_visibility_graph(c_space_obstacles, source, dest)
    # # TODO: fill in the next line
    # shortest_path, cost = None, None
    #
    # plotter3 = Plotter()
    # plotter3.add_robot(source, dist)
    # plotter3.add_obstacles(workspace_obstacles)
    # plotter3.add_robot(dest, dist)
    # plotter3.add_visibility_graph(lines)
    # plotter3.add_shorterst_path(list(shortest_path))
    #
    # plotter3.show_graph()
