import sys

sys.path.append('../../')
import numpy as np
import networkx as nx
import time
import utm


def astar_heuristic(a, b):
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)


class GridMap:
    def __init__(self, resolution, targets_gps, rover_gps, map_buffer=.1, obstacle_memory=16):
        """
        This class is used to create a map of the environment broken up into a grid (each square in the grid represents a
        resolution x resolution area of the environment). It can be used to find the shortest path from one point to another,
        with movement allowed in 8 directions (N, NE, E, etc.). The map is automatically fit to the target's GPS coordinates with an additonal buffer area

        :param resolution: The resolution of the grid (in meters). Each cell in the grid will be resolution x resolution meters
        :param targets_gps: A list of GPS coordinates for all the targets (in lat/lon format)
        :param rover_gps: The initial GPS coordinates of the rover (in lat/lon format)
        :param map_buffer: DEFAULT = 10%. The map is fit to the targets. This is a buffer added to each side of the map to ensure the rover doesn't go out of bounds
        :param obstacle_memory: DEFAULT = 16. If the rover sees an obstacle, it will remember it for this many iterations.
        This is to prevent it from remembering obstacles that it saw a long time ago and, due to odometry drift, may be somewhere else
        """

        self.resolution = resolution

        targets_utm = [utm.from_latlon(target[0], target[1])[:2] for target in targets_gps]
        min_utm_x, min_utm_y = min([target[0] for target in targets_utm]), min([target[1] for target in targets_utm])
        max_utm_x, max_utm_y = max([target[0] for target in targets_utm]), max([target[1] for target in targets_utm])

        x_diff = max_utm_x - min_utm_x
        y_diff = max_utm_y - min_utm_y

        # Add the buffer to the map
        min_utm_x, min_utm_y = min_utm_x - x_diff * map_buffer, min_utm_y - y_diff * map_buffer
        max_utm_x, max_utm_y = max_utm_x + x_diff * map_buffer, max_utm_y + y_diff * map_buffer

        # we only need the min or the max coordinates to convert UTM coordinates to grid coordinates
        self.min_utm_x = min_utm_x
        self.min_utm_y = min_utm_y

        # using the new min/max coordinates and the resolution, we can calculate the width and height of the map
        self.map_width = int((max_utm_x - min_utm_x) / resolution)
        self.map_height = int((max_utm_y - min_utm_y) / resolution)

        # convert the targets to grid coordinates
        self.targets = [self.gps_to_grid_coordinates(target[0], target[1]) for target in targets_gps]
        self.rover_position = self.gps_to_grid_coordinates(rover_gps[0], rover_gps[1])

        # now that we have the targets and the map height/width, we can create the map (as a networkx graph)
        self.map = self._create_map()


    def _create_map(self):
        G = nx.grid_2d_graph(self.map_width, self.map_height)

        # assign the edges their weight
        for u, v in G.edges:
            G.edges[u, v]['weight'] = self.resolution

        # the grid is made of squares, to the diagonal distance is higher than the horizontal/vertical distance
        diagonal_weight = np.sqrt(2) * self.resolution

        # Add diagonal edges and also the attributes for each node
        for node in G.nodes:
            G.nodes[node]['obstacle'] = False
            G.nodes[node]['obstacle_memory'] = 0
            G.nodes[node]['target'] = False

            x, y = node

            if x > 0:
                G.add_edge(node, (x-1, y))
                G.edges[node, (x-1, y)]['weight'] = diagonal_weight
            if x < self.map_width - 1:
                G.add_edge(node, (x+1, y))
                G.edges[node, (x+1, y)]['weight'] = diagonal_weight
            if y > 0:
                G.add_edge(node, (x, y-1))
                G.edges[node, (x, y-1)]['weight'] = diagonal_weight
            if y < self.map_height - 1:
                G.add_edge(node, (x, y+1))
                G.edges[node, (x, y+1)]['weight'] = diagonal_weight

        # Add the targets
        for target_x, target_y in self.targets:
            G.nodes[(target_x, target_y)]['target'] = True

        return G

    def _weight_fn(self, u, v, d):
        if self.map.nodes[v]['obstacle_memory'] > 0:
            return None # if we return none, the edge will be ignored
        else:
            return d['weight']

    def gps_to_grid_coordinates(self, lat, lon):
        """
        Converts GPS coordinates to grid coordinates.
        If the lat/lon is outside the map, it will return the closest grid cell
        """

        utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)

        diff_x, diff_y = utm_x - self.min_utm_x, utm_y - self.min_utm_y

        x, y = int(diff_x / self.resolution), int(diff_y / self.resolution)

        # cap the values to be within the map
        x = min(max(x, 0), self.map_width - 1)
        y = min(max(y, 0), self.map_height - 1)

        return x, y

    def get_path_to(self, lat, lon):
        """
        Returns the shortest path from the rover's current position to the target (in lat/lon format)

        :return: List of grid coordinates [(x1, y1), (x2, y2), ...] that the rover should follow, or None if the path doesn't exist
        """

        goal_x, goal_y = self.gps_to_grid_coordinates(lat, lon)

        try:
            path = nx.astar_path(self.map, self.rover_position, (goal_x, goal_y), heuristic=astar_heuristic,
                                 weight=self._weight_fn)
            return path
        except nx.exception.NetworkXNoPath:
            return None

    def update_rover(self, rover_lat, rover_lon):
        """
        Updates the rover's position on the map and decreases the obstacle memory for each grid cell
        """

        self.rover_position = self.gps_to_grid_coordinates(rover_lat, rover_lon)

        for node in self.map.nodes:
            if self.map.nodes[node]['obstacle_memory'] > 0:
                self.map.nodes[node]['obstacle_memory'] -= 1


    def update_obstacle(self):
        """TODO: figure out how we pass the location of obstacles to the grid map"""
        pass


if __name__ == '__main__':
    resolution = .5

    GPSList = [
        [37.3372215, -121.8819474],
        [37.3371282, -121.8819252],
        [37.3370621, -121.8818977],
        [37.3370451, -121.8818334],
        [37.3371197, -121.8818421],
        [37.3371991, -121.8818763],
        [37.3372471, -121.8818998],
        [37.337250, -121.881935],
    ]

    rover_gps = [37.337250, -121.881935]

    grid_map = GridMap(resolution, GPSList, rover_gps)
    path = grid_map.get_path_to(GPSList[0][0], GPSList[0][1])
    print(path)
    assert len(path) == 9, f'Path should be 9 nodes long, got {len(path)}, check to make sure the path is correct'

    # move the rover to the first lat/lon coordinate and get the path to the 2nd lat/lon coordinate
    grid_map.update_rover(GPSList[0][0], GPSList[0][1])
    path = grid_map.get_path_to(GPSList[1][0], GPSList[1][1])
    print(path)
    assert len(path) == 26, f'Path should be 26 nodes long, got {len(path)}, check to make sure the path is correct'

