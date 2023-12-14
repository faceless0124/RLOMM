import os.path as osp
import torch
import json
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, name):
        # parent_path = 'path'
        # self.MIN_LAT, self.MIN_LNG, self.MAX_LAT, self.MAX_LNG = utils.get_border(root_path + 'road.txt')
        if not path.endswith('/'):
            path += '/'
        self.data_path = osp.join(path, f"{name}_data/{name}.json")
        self.buildingDataset(self.data_path)

    def buildingDataset(self, data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)
            self.traces_ls = data[0::4]
            self.roads_ls = data[1::4]
            self.candidates = data[2::4]
            self.sampleIdx_ls = data[3::4]
        self.length = len(self.traces_ls)
        print(self.data_path)
        print(len(self.traces_ls), len(self.roads_ls), len(self.candidates), len(self.sampleIdx_ls))
        assert len(self.traces_ls) == len(self.roads_ls)
        assert len(self.traces_ls) == len(self.candidates)
        assert len(self.traces_ls) == len(self.sampleIdx_ls)

    def __getitem__(self, index):
        return self.traces_ls[index], self.roads_ls[index],\
            self.candidates[index], self.sampleIdx_ls[index]

    def __len__(self):
        return self.length

def padding(batch):
    trace_lens = [len(sample[0]) for sample in batch]
    # road_lens = [len(sample[1]) for sample in batch]
    candidates_lens = [len(candidates) for sample in batch for candidates in sample[2]]
    max_tlen, max_clen = max(trace_lens), max(candidates_lens)
    x, y, z, w = [], [], [], []
    # 0: [PAD]
    for sample in batch:
        x.append(sample[0] + [[0, 0]] * (max_tlen - len(sample[0])))
        y.append(sample[1] + [-1] * (max_tlen - len(sample[1])))
        z.append([candidates + [[0, 0]] * (max_clen-len(candidates)) for candidates in sample[2]] + [[[0, 0]] * max_clen] * (max_tlen - len(sample[2])))
        w.append([candidates_id + [-1] * (max_clen-len(candidates_id)) for candidates_id in sample[3]] + [[-1] * max_clen] * (max_tlen - len(sample[3])))
    print(torch.FloatTensor(x).shape, torch.LongTensor(y).shape, torch.FloatTensor(z).shape, torch.LongTensor(w).shape)
    return torch.FloatTensor(x), torch.LongTensor(y), torch.FloatTensor(z), torch.LongTensor(w), trace_lens