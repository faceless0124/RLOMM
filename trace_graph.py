import networkx as nx
import pickle
import os.path as osp
import torch


class TraceGraph():
    def __init__(self, data_path, device):
        # load trace graph
        if not data_path.endswith('/'):
            data_path += '/'
        trace_graph = nx.read_gml(data_path + 'trace_graph.gml', destringizer=int)
        self.num_grids = trace_graph.number_of_nodes()
        trace_pt_path = data_path + 'trace_graph_pt/'
        # 2*num_of_edges
        self.trace_weight = torch.load(trace_pt_path + 'inweight.pt').float().to(device)
        self.trace_in_edge_index = torch.load(trace_pt_path + 'in_edge_index.pt').to(device)
        self.trace_out_edge_index = torch.load(trace_pt_path + 'out_edge_index.pt').to(device)

        self.singleton_grid_mask = torch.load(trace_pt_path + 'singleton_grid_mask.pt').to(device)
        self.singleton_grid_location = torch.load(trace_pt_path + 'singleton_grid_location.pt').to(device)
        self.map_matrix = torch.load(trace_pt_path + 'map_matrix.pt').to(device)
        pkl_path = osp.join(data_path, 'used_pkl/')
        self.grid2traceid_dict = pickle.load(
            open(pkl_path + 'grid2traceid_dict.pkl', 'rb'))
        self.traceid2grid_dict = {
            v: k
            for k, v in self.grid2traceid_dict.items()
        }
