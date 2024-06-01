import re
from typing import Dict, Tuple
from math import radians, cos, sin, sqrt, atan2
import pickle
import sys
from tqdm import tqdm

def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371  # Radius of the earth in kilometers
    return R * c


def get_road_mid(path: str) -> Dict[int, Tuple[float, float]]:
    """
    Read beijing_road.txt and return a dictionary with link_id as key and
    the midpoint of the corresponding road segment as value.
    """
    cr = re.compile(r"(\d*)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\|(.*)")
    road_dict = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            data = cr.findall(line)
            if len(data) != 0:
                link_id, _, _, _, _, _, points, _ = data[0]

                points = [list(map(float, point.split(' '))) for point in points.split(',')]
                if len(points) > 1:
                    # Calculating midpoint
                    mid_lat = (points[0][0] + points[-1][0]) / 2
                    mid_lon = (points[0][1] + points[-1][1]) / 2
                    road_dict[int(link_id)] = (mid_lat, mid_lon)

    return road_dict


# Example usage (assuming the file 'beijing_road.txt' exists)
# road_dict = read_road("beijing_road.txt")

def calculate_distances(road_dict: Dict[int, Tuple[float, float]], city: str) -> Dict[Tuple[int, int], float]:
    """
    Calculate the distances between midpoints of road segments for each pair of link_ids.
    """
    distance_dict = {}
    link_ids = list(road_dict.keys())
    if city == 'beijing':
        link_cnt = 8533
    else:
        link_cnt = 4254

    for i in tqdm(range(len(link_ids))):
        for j in range(len(link_ids)):
            link_id1, link_id2 = link_ids[i], link_ids[j]
            coord1, coord2 = road_dict[link_id1], road_dict[link_id2]
            distance = haversine(coord1, coord2)
            distance_dict[(link_id1, link_id2)] = distance

    for j in range(len(link_ids)):
        link_id2 = link_ids[j]
        distance_dict[(link_cnt, link_id2)] = 100

    return distance_dict


if __name__ == '__main__':
    city = sys.argv[1]
    data_path = '../data/' + city + '/'
    road_dict = get_road_mid('../data/' + city + "_road.txt")
    dis_dict = calculate_distances(road_dict, city)
    pickle.dump(dis_dict, open(data_path + 'real_distances.pkl', 'wb'))