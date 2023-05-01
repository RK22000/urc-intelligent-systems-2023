import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from queue import PriorityQueue
from matplotlib.patches import Arrow
import random
import utm




class GridMapSimulator:
    def __init__(self, resolution, map_width, map_height, targets, init_gps,
                 lidar_range=1, num_initial_obstacles=20, obstacle_memory=4, interval=200, path_always_exists=True):
        self.resolution = resolution
        self.map_width = map_width
        self.map_height = map_height
        self.interval = interval
        self.ani = None
        self.path_plot = None  # Add an attribute to store the path plot

        self.targets = targets
        self.map = self._create_map()
        self.current_target_index = 0
        self.target_x, self.target_y = targets[self.current_target_index]

        self.reached_destination = False
        self.rover_direction = 0

        # Set the initial position based on the provided GPS coordinate
        init_lon, init_lat = init_gps
        self.rover_x, self.rover_y = gps_to_grid_coordinates(init_lat, init_lon, min_utm_x, min_utm_y, max_utm_x, max_utm_y)

        self.lidar_range = lidar_range
        # this flag determines whether we ensure a path always exists to the target when we generate obstacles (WARNING: it slows down obstacle generation by several seconsd)
        self.path_always_exists = path_always_exists
        self.generate_initial_obstacles(num_initial_obstacles)
        self.obstacle_memory = obstacle_memory
        self.detect_obstacle()


    def _create_map(self):
        G = nx.grid_2d_graph(self.map_width, self.map_height)

        # Add diagnol edges and also the attributes for each node
        for node in G.nodes:
            G.nodes[node]['obstacle'] = False
            G.nodes[node]['obstacle_memory'] = 0
            G.nodes[node]['target'] = False

            x, y = node

            if x > 0:
                G.add_edge(node, (x-1, y))
            if x < self.map_width - 1:
                G.add_edge(node, (x+1, y))
            if y > 0:
                G.add_edge(node, (x, y-1))
            if y < self.map_height - 1:
                G.add_edge(node, (x, y+1))

        # Add the targets
        for target_x, target_y in self.targets:
            G.nodes[(target_x, target_y)]['target'] = True

        return G

    def _map_2_img(self):
        map_img = np.ones((self.map_height, self.map_width), dtype=np.float32)

        # the background is white, obstacles that haven't been seen are light gray, and obstacles that have been seen are black (but get lighter over time)
        for node in self.map.nodes:
            x, y = node
            if self.map.nodes[node]['obstacle']:
                map_img[y, x] = .5 - (self.map.nodes[node]['obstacle_memory'] / self.obstacle_memory) * .5

        return map_img

    def generate_initial_obstacles(self, num_obstacles):
        print(f'Generating {num_obstacles} obstacles...')
        cur_obstacles = 0
        while cur_obstacles < num_obstacles:
            x = np.random.randint(0, self.map_width)
            y = np.random.randint(0, self.map_height)

            # We randomly generate an obstacle, but we also have to make sure that it's not on top of the rover, it's target, or another obstacle
            if not self.map.nodes[(x, y)]['obstacle'] and not self.map.nodes[(x, y)]['target'] and not (x == self.rover_x and y == self.rover_y):
                self.map.nodes[(x, y)]['obstacle'] = True
                cur_obstacles += 1

            if self.path_always_exists:
                # Also, we want to make sure that we have a path to each target, although this code drastically slows down the simulation creation, not sure how to make it faster
                for target_x, target_y in self.targets:
                    path = self.find_path(self.rover_x, self.rover_y, target_x, target_y)
                    if path is None:
                        self.map.nodes[(x, y)]['obstacle'] = False
                        cur_obstacles -= 1
                        break

    def init_visualization(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Grid Map")
        self.path_plot, = self.ax.plot([], [], color='red', linestyle='-')
        self.path_line, = self.ax.plot([], [], color='lime')


        self.ani = FuncAnimation(self.fig, animate, fargs=(self,), interval=self.interval)
        self.grid_img = self.ax.imshow(self._map_2_img(), origin='lower', cmap='gray', vmin=0, vmax=1)

        # Add the initial position of the rover as a red dot
        arrow_length = 1
        arrow_width = 2
        self.rover_arrow = Arrow(self.rover_x - arrow_length / 2, self.rover_y - arrow_width / 2, arrow_length, arrow_width, color='gray', zorder=2)
        self.ax.add_patch(self.rover_arrow)

        # Add the destination positions as blue dots and number them
        self.target_dots = []
        for i, (target_x, target_y) in enumerate(self.targets):
            target_dot = self.ax.scatter(target_x, target_y, c='blue')
            self.ax.text(target_x + 0.5, target_y + 0.5, str(i + 1), fontsize=12, color='blue')
            self.target_dots.append(target_dot)


    def update_visualization(self, target_x, target_y):
        # Find the optimal path from the current position to the target position using A*
        path = self.find_path(self.rover_x, self.rover_y, target_x, target_y)
        if path is not None:
            path_x, path_y = zip(*path)
            self.path_line.set_data(path_x, path_y)

        arrow_length = 1
        arrow_width = 1
        self.rover_arrow.remove()

        x_offset, y_offset = 0.5, 0.5
        self.rover_arrow = Arrow(self.rover_x + 0.5 - x_offset,
                                 self.rover_y + 0.5 - y_offset,
                                 arrow_length * np.cos(self.rover_direction),
                                 arrow_length * np.sin(self.rover_direction),
                                 color='gray', zorder=2)

        self.ax.add_patch(self.rover_arrow)


        self.grid_img.set_data(self._map_2_img())

        # Check if the rover has reached the target position
        if self.rover_x == target_x and self.rover_y == target_y:
            print("I have made it to the destination!")
            # plt.close(self.fig)  # Stop the animation
            # exit(1)

        return [self.grid_img, self.rover_arrow, *self.target_dots, self.path_plot]


    def find_path(self, start_x, start_y, goal_x, goal_y):
        try:
            path = nx.astar_path(self.map, (start_x, start_y), (goal_x, goal_y), heuristic=astar_heuristic,
                                 weight=lambda u, v, d: None if self.map.nodes[v]['obstacle'] else 1) # if the weight is None, then astar treats the edge as untraversable
            return path
        except nx.exception.NetworkXNoPath:
            return None


    def detect_obstacle(self):
        # reduce the memory of all the obstacles currently seen
        for node in self.map.nodes:
            if self.map.nodes[node]['obstacle']:
                self.map.nodes[node]['obstacle_memory'] = max(self.map.nodes[node]['obstacle_memory'] - 1, 0)

        # now, simulate checking for obstacles in the 8 directions around the rover (up to the lidar range)
        for i in range(max(0, self.rover_x - self.lidar_range), min(self.map_width, self.rover_x + self.lidar_range)):
            for j in range(max(0, self.rover_y - self.lidar_range), min(self.map_height, self.rover_y + self.lidar_range)):
                if self.map.nodes[(i, j)]['obstacle']:
                    # if there is an obstacle, update the obstacle memory
                    self.map.nodes[(i, j)]['obstacle_memory'] = self.obstacle_memory



    def move_rover(self):
        if self.reached_destination:
            return
        # Detect obstacles before moving
        self.detect_obstacle()

        target_x, target_y = self.targets[self.current_target_index]

        # Find the optimal path from the current position to the target position using A*
        path = self.find_path(self.rover_x, self.rover_y, target_x, target_y)
        if path is None or len(path) < 2:
            # If there is no path or the path is too short, do not move the rover
            print("No path found or path too short")
            return

        # Move the rover one step along the optimal path
        new_x, new_y = path[1]

        dx, dy = new_x - self.rover_x, new_y - self.rover_y
        new_direction = np.arctan2(dy, dx)
        if new_direction != self.rover_direction:
            self.rover_direction = new_direction
            print(f"Turned to angle {np.degrees(self.rover_direction)}")

        print("Moved to position ({}, {})".format(new_x, new_y))
        self.rover_x, self.rover_y = new_x, new_y

        # Add the new position to the path plot
        path_x, path_y = self.path_plot.get_data()
        path_x = np.append(path_x, self.rover_x)
        path_y = np.append(path_y, self.rover_y)
        self.path_plot.set_data(path_x, path_y)

        # Check if the rover has reached the target position
        if self.rover_x == target_x and self.rover_y == target_y:
            print("I have made it to the destination!")
            self.reached_destination = True
            if self.current_target_index + 1 < len(self.targets):
                self.current_target_index += 1
                self.reached_destination = False
                print(f"Moving to the next target: {self.targets[self.current_target_index]}")
            else:
                print("All targets reached!")

def astar_heuristic(a, b):
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)


# Example usage
def animate(frame, grid_map, *args):
    grid_map.move_rover()
    target_x, target_y = grid_map.targets[grid_map.current_target_index]
    return grid_map.update_visualization(target_x, target_y)

def generate_random_targets(num_targets, map_width, map_height):
    random_targets = []
    for _ in range(num_targets):
        x = random.randint(0, map_width - 1)
        y = random.randint(0, map_height - 1)
        random_targets.append((x, y))
    return random_targets


def gps_to_grid_coordinates(lat, lon, min_utm_x, min_utm_y, max_utm_x, max_utm_y):
    utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)
    normalized_x = (utm_x - min_utm_x) / (max_utm_x - min_utm_x)
    normalized_y = (utm_y - min_utm_y) / (max_utm_y - min_utm_y)
    x = int((normalized_x * (map_width - 1)) + 0.5)
    y = int((normalized_y * (map_height - 1)) + 0.5)
    return x, y

# Made from https://www.gpsvisualizer.com/draw/
GPSList = [
    [-121.8819474, 37.3372215],
    [-121.8819252, 37.3371282],
    [-121.8818977, 37.3370621],
    [-121.8818334, 37.3370451],
    [-121.8818421, 37.3371197],
    [-121.8818763, 37.3371991],
    [-121.8818998, 37.3372471],
    [-121.881935, 37.337250],
]

map_width = 25
map_height = 25
coordinate_list = []

# Convert GPS to UTM and find the min and max UTM coordinates
utm_coords = [utm.from_latlon(lat, lon)[:2] for lon, lat in GPSList]
min_utm_x, min_utm_y = map(min, zip(*utm_coords))
max_utm_x, max_utm_y = map(max, zip(*utm_coords))

for i in range(len(GPSList)):
    lon, lat = GPSList[i]
    x, y = gps_to_grid_coordinates(lat, lon, min_utm_x, min_utm_y, max_utm_x, max_utm_y)
    coordinate_list.append((x, y))
    print("grid point", x, y)




resolution = 1
lidar_range = 2 # this is how many squares away the rover can see an obstacle
map_width = 30
map_height = 30
initial_obstacles = 300
obstacle_memory = 16 # this is the number of frames that an obstacle is remembered/included in the astar search after it was detected.
animation_speed = 100
num_targets = 3

init_gps = [-121.881935, 37.337250]
grid_map = GridMapSimulator(resolution, map_width, map_height, coordinate_list, init_gps,
                            lidar_range=lidar_range,
                            num_initial_obstacles=initial_obstacles,
                            obstacle_memory=obstacle_memory,
                            interval=animation_speed)
target_x, target_y = coordinate_list[grid_map.current_target_index]
grid_map.init_visualization()
plt.show()