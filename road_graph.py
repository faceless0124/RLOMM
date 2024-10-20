from collections import deque

import networkx as nx
import torch
from torch_sparse import SparseTensor
import pickle
import os.path as osp

from tqdm import tqdm


class RoadGraph():
    def __init__(self, root_path, layer, gamma, device) -> None:
        self.device = device
        self.root_path = root_path
        # load road graph
        if not root_path.endswith('/'):
            root_path += '/'
        road_graph = pickle.load(open(root_path + 'road_graph.pkl', 'rb'))
        self.num_roads = road_graph.number_of_nodes()
        # load edge weight of road graph
        road_pt_path = root_path + 'road_graph_pt/'
        # 2*num_of_edges
        road_edge_index = torch.load(road_pt_path + 'edge_index.pt')
        # construct sparse adj
        self.road_adj = SparseTensor(row=road_edge_index[0],
                                     col=road_edge_index[1],
                                     sparse_sizes=(self.num_roads,
                                                   self.num_roads)).to(device)
        self.road_x = torch.load(road_pt_path + 'x.pt').float().to(device)

        connectivity_distances_file = root_path + 'connectivity_distances.pkl'
        with open(connectivity_distances_file, 'rb') as f:
            self.connectivity_distances = pickle.load(f)
        # real_distances_file = root_path + 'real_distances.pkl'
        # with open(real_distances_file, 'rb') as f:
        #     self.real_distances = pickle.load(f)

        # self.connectivity_distances = None
        self.real_distances = None

    def precompute_distances(self):
        self.road_adj_coo = self.road_adj.to('cpu').coo()
        rows, cols, _ = self.road_adj_coo
        self.neighbors = {i: cols[rows == i].tolist() for i in range(self.num_roads)}
        distances = {}

        for road_id1 in tqdm(range(self.num_roads)):
            queue = deque([road_id1])
            visited = set([road_id1])
            distance = {road_id1: 0}

            while queue:
                current_id = queue.popleft()

                for neighbor_id in self.neighbors[current_id]:
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
                        visited.add(neighbor_id)
                        distance[neighbor_id] = distance[current_id] + 1

            for road_id2 in range(self.num_roads):
                if road_id1 != road_id2:
                    distances[(road_id1, road_id2)] = distance.get(road_id2, -1)

        with open(self.root_path + '/connectivity_distances.pkl', 'wb') as f:
            pickle.dump(distances, f)

        return distances


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    road_graph = RoadGraph(root_path='./data/beijing',
                           layer=4,
                           gamma=10000,
                           device=device)
    road_graph.precompute_distances()
