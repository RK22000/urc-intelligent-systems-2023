import sys
sys.path.append( '../../')
from Autonomous_Systems.ObjectDetector import ObjectDetector

if __name__ == "__main__":
    # change the `lidar_port` parameter to the port being used on my machine
    # how to check on macos: 
    # Plug in device and run:  ls -lha /dev/tty* > plugged.txt
    # Unplug device and run:   ls -lha /dev/tty* > np.txt
    # Compare files:           diff plugged.txt np.txt
    object_detector = ObjectDetector(lidar_port='/dev/ttys002', VISUALIZE=False, MaxDistance=4.0)
    for camera_boxes, lidar_data in object_detector.calculate_objects():
        print("Position:", camera_boxes, "Distance:", lidar_data, "meters")
