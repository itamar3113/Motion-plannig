import argparse
import os
from typing import List, Tuple
import numpy as np
from Plotter import Plotter
from shapely.geometry.polygon import Polygon, LineString, Point


# TODO remove in is not needed
def sort_vertices_counterclockwise(polygon):
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
    robot_in_origin = [(0, -r), (r, 0), (0, r), (-r, 0)]
    res = []
    for p1 in vertices:
        for p2 in robot_in_origin:
            res.append((p1[0] + p2[0], p1[1] + p2[1]))
    return Polygon(res).convex_hull
    #T ODO check about comlexity and using library functions.

    # other way to compute minkowsky sum
    # i, j = 0, 0
    # res = []
    # len_poly = len(vertices)
    # len_robot = len(robot_in_origin)
    # p1, next_poly = vertices[i]
    # p2, next_robot = robot_in_origin[j]
    # while i < len_poly or j < len_robot:
    #     p1 = vertices[i % len_poly]
    #     p2 = robot_in_origin[j % len_robot]
    #     res.append((p1[0] + p2[0], p1[1] + p2[1]))
    #     cross = cross_product((vertices[(i + 1) % len_poly][0] - vertices[i % len_poly][0],
    #                            vertices[(i + 1) % len_poly][1] - vertices[i % len_poly][1]),
    #                           (robot_in_origin[(j + 1) % len_robot][0] - robot_in_origin[j % len_robot][0],
    #                            robot_in_origin[(j + 1) % len_robot][1] - robot_in_origin[j % len_robot][1]))
    #     if cross >= 0:
    #         i += 1
    #     if cross <= 0:
    #         j += 1
    # return Polygon(res)


# TODO
def get_visibility_graph(obstacles: List[Polygon], source=None, dest=None) -> List[LineString]:
    """
    Get The visibility graph of a given map
    :param obstacles: A list of the obstacles in the map
    :param source: The starting position of the robot. None for part 1.
    :param dest: The destination of the query. None for part 1.
    :return: A list of LineStrings holding the edges of the visibility graph
    """
    res = []
    obstacles_edges = []
    # loop over all the obstacles and add their edges to the map.
    for obstacle in obstacles:
        vertices = obstacle.exterior.coords[:]
        j = 0
        edge = LineString([vertices[j], vertices[j + 1]])
        while j < len(vertices) - 1:
            edge = LineString([vertices[j], vertices[j + 1]])
            res.append(edge)
            obstacles_edges.append(edge)
            j += 1
    # add start and end points if exist
    if source is not None and dest is not None:
        obstacles.insert(0, source)
        obstacles.insert(1, dest)
    # loop over all vertices in obstacles
    for i in range(len(obstacles)):
        if source is not None and i < 2:
            vertices = [obstacles[i]]
        else:
            vertices = obstacles[i].exterior.coords[:-1]
        for vertex in vertices:
            # check for all the vertexes in different polygons if they have visibility line.
            k = i + 1
            while k < len(obstacles):
                if source is not None and k == 1:
                    other_vertices = [obstacles[k]]
                else:
                    other_vertices = obstacles[k].exterior.coords[:-1]
                k += 1
                for other_vertex in other_vertices:
                    edge = LineString([vertex, other_vertex])
                    is_visible = True
                    # check if the edge intersect with one of the obstacles edges.
                    for obstacle_edge in obstacles_edges:
                        # if the edges have a common edge skip it.
                        if vertex in obstacle_edge.coords or other_vertex in obstacle_edge.coords:
                            continue
                        if edge.intersects(obstacle_edge):
                            is_visible = False
                            break
                    if is_visible:
                        res.append(edge)
    return res


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

    #plotter1.show_graph()

    # step 2:

    lines = get_visibility_graph(c_space_obstacles)
    plotter2 = Plotter()

    plotter2.add_obstacles(workspace_obstacles)
    plotter2.add_c_space_obstacles(c_space_obstacles)
    plotter2.add_visibility_graph(lines)
    plotter2.add_robot(source, dist)

    plotter2.show_graph()

    # # step 3:
    with open(query, 'r') as f:
        dest = tuple(map(float, f.readline().split(',')))

    lines = get_visibility_graph(c_space_obstacles, source, dest)
    # TODO: fill in the next line
    shortest_path, cost = None, None

    plotter3 = Plotter()
    plotter3.add_robot(source, dist)
    plotter3.add_obstacles(workspace_obstacles)
    plotter3.add_robot(dest, dist)
    plotter3.add_visibility_graph(lines)
    #plotter3.add_shorterst_path(list(shortest_path))

    plotter3.show_graph()
