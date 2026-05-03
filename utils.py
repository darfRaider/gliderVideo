import numpy as np
from gpxpy.gpx import GPXTrackPoint
from dataclasses import dataclass


@dataclass
class Bearing:
    bearing_rad: float
    bearing_deg: float
    climb_rate: float

def get_bearing(pt1: GPXTrackPoint, pt2: GPXTrackPoint) -> Bearing:
    assert pt1.time < pt2.time, f"Track point 2 must contain more recent data than point 1 ({pt1}, {pt2})"
    lat1, lon1, lat2, lon2 = map(np.radians, [pt1.latitude, pt1.longitude, 
                                              pt2.latitude, pt2.longitude])
    d_lon = lon2 - lon1
    
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)

    bearing_rad = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
    bearing_deg = np.degrees(bearing_rad)
    climb_rate = (pt2.elevation - pt1.elevation) / (pt2.time - pt1.time).total_seconds()

    return Bearing(bearing_rad, bearing_deg, climb_rate)
