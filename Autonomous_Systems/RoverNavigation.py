"""
Rover Navigation 


Returns:
    _type_: _description_
"""




import os, sys
sys.path.insert(0, os.path.abspath(".."))
from queue import PriorityQueue
from Autonomous_Systems import AutoHelp
from simple_pid import PID
import time
import numpy as np
import json
from math import atan2, sqrt, pi
from numpy import linalg as la
from Autonomous_Systems.GridMap import GridMap

class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0.1, 0],
                           [0, 0, 0, 0.1]])
        self.R = np.array([[10, 0],
                           [0, 10]])
        self.P = np.eye(self.A.shape[0]) * 1000

    def predict(self, x):
        x = np.dot(self.A, x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return x

    def update(self, x, z):
        y = z - np.dot(self.C, x)
        S = np.dot(np.dot(self.C, self.P), self.C.T) + self.R
        K = np.dot(np.dot(self.P, self.C.T), la.inv(S))
        x = x + np.dot(K, y)
        self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.C)), self.P)
        return x

class RoverNavigation:
    def __init__(self, max_speed, max_steering, GPS, IMU, GPS_coordinate_map, map_resolution=.5):
        self._initialize_constants(max_speed,max_steering)
        self._initialize_sensors(GPS,IMU,GPS_coordinate_map)
        self._initialize_controllers()
        self._initialize_mapping(map_resolution)

    def _initialize_constants(self, max_speed, max_steering):
        """Initialize constants that are tied to Rover Controls

        Args:
            max_speed (_type_): _description_
            max_steering (_type_): _description_
        """
        self.max_speed = max_speed
        self.max_steering = max_steering
        self.commands = [0,1,0,'D',0,0]


    def _initialize_sensors(self, GPS, IMU, GPS_coordinate_map):
        """ Initialize sensors and the kalman filter that will tie their readings together

        """
        self.GPS = GPS
        self.GPS_coordinate_map = GPS_coordinate_map
        self.GPS_target = self.GPS_coordinate_map[0]

        self.IMU = IMU
        self.AutoHelp = AutoHelp.AutoHelp()
        self.filter = KalmanFilter(dt=0.1)
    
    def _initialize_mapping(self, resolution):
        """ Initialize the map that will be used to track the rover's current position
         
        """
        self.map = GridMap(resolution, self.GPS_coordinate_map, self.GPS.get_position())


    def _initialize_controllers(self):
        """
            Initialize the controls algorithms.
        """
        # Create a PID controller for steering
        self.steer_controller = PID(Kp=1, Ki=0.5, Kd=0.05, setpoint=0)
        self.steer_controller.sample_time = 0.1
        self.steer_controller.output_limits = (-self.max_steering, self.max_steering)
        self.steer_controller.proportional_on_measurement = True  # Use derivative of error instead of error for Kp

        # Create a PID controller for speed
        self.speed_controller = PID(Kp=0.5, Ki=0.1, Kd=0.05, setpoint=self.max_speed)
        self.speed_controller.sample_time = 0.1
        self.speed_controller.output_limits = (0, self.max_speed)



    def PID_steer(self, commands, steer_output, angle):
        """Steer the rover using a PID controller"""

        speed_error = self.max_steering/abs(steer_output)   # scale speed error based on steering output
        print("Speed Error:", speed_error)
        speed_output = self.speed_controller(speed_error)
        self.commands[4] = round(speed_output)
        if angle == "right":
            self.commands[5] = abs(round(steer_output))
        elif angle == "left":
            self.commands[5] = -abs(round(steer_output))
        return self.AutoHelp.jsonify_commands(commands)


    def forward_drive(self, commands):
        """Drive the rover forward at max speed"""

        self.commands[4] = self.max_speed
        self.commands[5] = 0
        return self.AutoHelp.jsonify_commands(commands)

    def spin(self, commands, angle):
        self.commands[5] = 0
        if angle == "right":
            self.commands[4] = abs(round(self.max_speed/2))
        elif angle == "left":
            self.commands[4] = -abs(round(self.max_speed/2))
        
        return self.AutoHelp.jsonify_commands(commands)

    def stop_rover(self, commands):
        self.commands = [0, 0, 0, 'D', 0, 0]
        return self.AutoHelp.jsonify_commands(commands)

    def goto_next_coordinate(self):
        try:
            self.GPS_coordinate_map.pop(0)
            self.GPS_target = self.GPS_coordinate_map[0]
            print("Going to new coordinate!")
            print("NEW:", self.GPS_target)
        except:
            print("No more GPS coordniates in Mission... Mission Success!")
            exit(1)
            
    def path_to_command(self, path, current_heading):
        """Takes a path (list of (x, y) coordinates and turns it into a command for the rover.
        If the path is too short (only one coordinate), returns the Stop command"""

        command = []
        if path is not None and len(path) > 1:
            # we can only follow one command at a time, and after each command is completed, the path may change, so only look at the next one
            start_x, start_y = path[0]
            end_x, end_y = path[1]

            # Calculate the angle and distance between the start and end points
            dx = end_x - start_x
            dy = end_y - start_y

            angle = np.arctan2(dy, dx)
            change_in_angle = int(np.degrees(angle) - current_heading) # rover expects angle in degrees and integers

            # Set the speed and angle to default values
            speed = 1

            # Set the drive mode based on the direction of the movement
            if change_in_angle < 150:
                mode = 'D'
            else:
                mode = 'S'

            command.append(mode)
            command.append(speed)
            command.append(change_in_angle)

        return command


    def move_rover(self):
        #TODO Needs to implent the compass/IMU and GPS to move the rover towards the proper direction and location


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
            print(f"Turned to angle {np.degrees(self.rover_direction)}") #TODO translate this to our actual rover angles

        print("Moved to position ({}, {})".format(new_x, new_y))
        self.map[self.rover_y, self.rover_x] = 0  # Clear the old rover's position
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
        


# THIS SHOULD BE USED TO HAVE THE ROVER GO INTO DIFFERENT MODES SUCH AS SPIN OR TRANSLATE
# def get_steering(self, current_GPS, GPS_target):
#         quat_i, quat_j, quat_k, quat_real = self.IMU.get_rotation()
#         rover_heading = self.IMU.get_heading(quat_real, quat_i, quat_j, quat_k)
#         bearing = self.AutoHelp.get_bearing(current_GPS, GPS_target)
#         self.steer_controller.setpoint= bearing
#         distance = round(self.AutoHelp.get_distance(current_GPS, GPS_target)[0]*1000, 3)
#         direction = round((bearing - rover_heading + 360) % 360, 3)

#         steer_error = self.steer_controller(rover_heading)
#         print("Direction:", direction)

#         if distance <= 3:
#             print("Arrived at target!")
#             self.goto_next_coordinate()
#             time.sleep(3)
#             return self.stop_rover(self.commands)

#         if abs(direction) > 15:

#             if direction >= 150 and direction <= 180:
#                 print("Going to Spin Mode Right")
#                 self.change_modes('S')
#                 return self.spin(self.commands, 'right')
#             elif direction >= 180 and direction <= 210:
#                 print("Going to Spin Mode Left")
#                 self.change_modes('S')
#                 return self.spin(self.commands, 'left')

#             if self.commands[3] == 'S' and direction < 30 or direction > 330:
#                 self.change_modes('D')

#             if self.commands[3] == 'D' and direction < 150:
#                 print("Turning right")
#                 return self.PID_steer(self.commands, steer_error, 'right')
#             elif self.commands[3] == 'D' and direction > 210:
#                 print("Turning left")
#                 return self.PID_steer(self.commands, steer_error, 'left')
#         rover_heading = bearing
        
#         if direction <= 15:
#             print("Moving forward")
#             self.change_modes('D')
#             return self.forward_drive(self.commands)
