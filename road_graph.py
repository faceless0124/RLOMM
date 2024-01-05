import networkx as nx
import torch
from torch_sparse import SparseTensor
import pickle
import os.path as osp


class RoadGraph():
    def __init__(self, root_path, layer, gamma, device) -> None:
        self.device = device
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
        # gain A^k
        A = torch.load(road_pt_path + 'A.pt')
        # A_list [n, n]
        self.A_list = self.get_adj_poly(A, layer, gamma)

    def get_adj_poly(self, A, layer, gamma):
        A_ = A.to(self.device)
        ans = A_.clone()
        for _ in range(layer - 1):
            ans = ans @ A_
        ans[ans != 0] = 1.
        ans[ans == 0] = -gamma
        return ans


