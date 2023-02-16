import os, sys
sys.path.insert(0, os.path.abspath(".."))
import requests
import serial.tools.list_ports as port_list
from modules.Serial import SerialSystem
from CommandScripts.autonomy import Autonomy
from modules.GPS import gpsRead
import json

serial_port = "/dev/ttyACM0"
gps_port = "/dev/ttyACM2"
baudrate = 38400
max_speed = 5
max_angle = 12
server = 'http://10.251.253.243:5002'
GPS_list = []

try:
    serial = SerialSystem(serial_port, baudrate)
    print("Using port: " + serial_port, "For Serial Comms")
except:
    ports = list(port_list.comports())
    print('====> Designated Port not found. Using Port:', ports[0].device, "For Serial Connection")
    serial_port = ports[0].device
    serial = SerialSystem(serial_port, baudrate)


try:

    GPS = gpsRead(gps_port,9600)
    print("Using port: " + gps_port, "For GPS")
except:
        port_number = 0
        ports = list(port_list.comports())
        print('====> Designated Port not found. Using Port:', ports[port_number].device, "For GPS Connection")
        port = ports[port_number].device
        GPS = gpsRead(port,9600)
        while GPS.get_position() == ['error', 'error'] or GPS.get_position() == ["None", "None"]:
            print("Port not found, going to next port...")
            port_number += 1
            gps_port = ports[port_number].device
            try:
                GPS = gpsRead(port,9600)
            except:
                continue
            break


GPS_map_url = f"{server}/gps_map"

try:
    GPS_map = requests.get(GPS_map_url)
except:
    print("Could not get GPS map from mission control")
    exit(1)

GPS_map = json.loads(GPS_map.text)

for i in GPS_map:
    GPS_list.append(GPS_map[i])
print("GPS List:", GPS_list)

rover = Autonomy(serial, server, max_speed, max_angle, GPS, GPS_list)
rover.start_mission()
