import serial
import requests
import math

class gpsRead:
    def __init__(self, port, baudrate):
        self.gps_port = serial.Serial(port, baudrate)

    def send_request(self, LonLat, url):
        LonLat = list(LonLat)
        requests.post(url, json={"longitude": LonLat["longitude"], "latitude": LonLat["latitude"]})

    def get_position(self, url=None):
        LonLat = {"longitude":0,
                  "latitude":0}
        try:
            line = self.gps_port.readline().decode('utf-8').strip()
            if line.startswith('$GNRMC'):
                parts = line.split(',')
                if parts[2] == 'A':
                    latitude = float(parts[3][:2]) + float(parts[3][2:]) / 60
                    if parts[4] == 'S':
                        latitude *= -1
                    longitude = float(parts[5][:3]) + float(parts[5][3:]) / 60
                    if parts[6] == 'W':
                        longitude *= -1
                    LonLat["longitude"] = longitude
                    LonLat["latitude"]= latitude
                    if url is not None:
                        self.send_request(LonLat, url)
                    return [longitude, latitude]
        except:
            return ['error', 'error']
