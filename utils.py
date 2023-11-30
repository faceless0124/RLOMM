import torch
import numpy as np
import math
import os

GRID_SIZE = 50
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05
DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS
EARTH_MEAN_RADIUS_METER = 6371008.7714
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER

def get_border(path):
    """
        get the min(max) LAT(LNG)
    """
    MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = 360, 360, -360, -360
    with open(path, 'r') as f:
        road_ls = f.readlines()
    for road in road_ls:
        tmpa = road.split('\t')[6].split('|')[0]
        lng_lat_ls = tmpa.split(',')
        for lng_lat in lng_lat_ls:
            lng = float(lng_lat.split(' ')[0])
            lat = float(lng_lat.split(' ')[1])
            MAX_LAT = max(MAX_LAT, lat)
            MAX_LNG = max(MAX_LNG, lng)
            MIN_LAT = min(MIN_LAT, lat)
            MIN_LNG = min(MIN_LNG, lng)
    return MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG