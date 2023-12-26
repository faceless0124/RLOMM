import json
import os.path as osp
from torch.utils.data import Dataset
import torch
import data_preprocess.utils as utils
import time

class MyDataset(Dataset):
    def __init__(self, path, name):
        if not path.endswith('/'):
            path += '/'
        self.data_path = osp.join(path, f"{name}_data/{name}.json")
        self.MIN_LAT, self.MIN_LNG, self.MAX_LAT, self.MAX_LNG = utils.get_border('./data/road.txt')
        self.buildingDataset(self.data_path)


    def buildingDataset(self, data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)
            self.traces_ls = data[0::5]
            self.time_stamps = data[1::5]
            self.roads_ls = data[2::5]
            self.candidates = data[3::5]
            self.candidates_id = data[4::5]
        self.length = len(self.traces_ls)

        # 归一化轨迹和候选点数据
        self.traces_ls = [self.normalize_traces(trace) for trace in self.traces_ls]
        self.candidates = [self.normalize_candidates(candidate) for candidate in self.candidates]
        self.time_stamps = [self.normalize_time_stamps(ts) for ts in self.time_stamps]

        self.cnt = self.count_point()

        print(self.data_path)
        print("Total points: ", self.cnt)
        print(len(self.traces_ls), len(self.time_stamps), len(self.roads_ls), len(self.candidates), len(self.candidates_id))
        assert len(self.traces_ls) == len(self.roads_ls)
        assert len(self.traces_ls) == len(self.candidates)
        assert len(self.traces_ls) == len(self.candidates_id)
        assert len(self.traces_ls) == len(self.time_stamps)


    def count_point(self):
        count = 0
        for trace in self.traces_ls:
            for point in trace:
                count += 1
        return count

    def normalize_traces(self, trace):
        # 归一化经纬度数据
        return [[(lat - self.MIN_LAT) / (self.MAX_LAT - self.MIN_LAT),
                 (lng - self.MIN_LNG) / (self.MAX_LNG - self.MIN_LNG)]
                for lat, lng in trace]

    def normalize_candidates(self, candidates):
        # 归一化候选点的经纬度数据
        normalized_candidates = []
        for candidate in candidates:
            normalized_candidate = []
            for point in candidate:
                lng, lat = point
                normalized_lat = (lat - self.MIN_LAT) / (self.MAX_LAT - self.MIN_LAT)
                normalized_lng = (lng - self.MIN_LNG) / (self.MAX_LNG - self.MIN_LNG)
                normalized_candidate.append([normalized_lat, normalized_lng])
            normalized_candidates.append(normalized_candidate)
        return normalized_candidates

    def normalize_time_stamps(self, time_stamps):
        # 将第一个时间点转换为秒
        base_time = time.mktime(time.strptime(time_stamps[0], '%Y/%m/%d %H:%M:%S'))
        # 计算每个时间点与第一个时间点的时间间隔
        normalized_time_stamps = [time.mktime(time.strptime(ts, '%Y/%m/%d %H:%M:%S')) - base_time for ts in time_stamps]
        return normalized_time_stamps

    def __getitem__(self, index):
        return self.traces_ls[index], self.time_stamps[index], self.roads_ls[index],\
            self.candidates[index], self.candidates_id[index]

    def __len__(self):
        return self.length


    def get_data(self, index):
        if index >= self.length:
            raise IndexError("索引超出数据集范围")
        return self.__getitem__(index)

    def get_length(self):
        return self.length

def padding(batch):
    trace_lens = [len(sample[0]) for sample in batch]
    # road_lens = [len(sample[1]) for sample in batch]
    candidates_lens = [len(candidates) for sample in batch for candidates in sample[3]]
    max_tlen, max_clen = max(trace_lens), max(candidates_lens)
    traces, time_stamp, roads, candidates, candidates_id = [], [], [], [], []
    # 0: [PAD]
    for sample in batch:
        traces.append(sample[0] + [[0, 0]] * (max_tlen - len(sample[0])))
        time_stamp.append(sample[1] + [-1] * (max_tlen - len(sample[1])))
        roads.append(sample[2] + [-1] * (max_tlen - len(sample[2])))
        candidates.append([candidates + [[0, 0]] * (max_clen-len(candidates)) for candidates in sample[3]] + [[[0, 0]] * max_clen] * (max_tlen - len(sample[3])))
        candidates_id.append([candidates_id + [8533] * (max_clen-len(candidates_id)) for candidates_id in sample[4]] + [[-1] * max_clen] * (max_tlen - len(sample[4])))
    traces = torch.cat((torch.FloatTensor(traces), torch.FloatTensor(time_stamp).unsqueeze(-1)), dim=-1)
    return traces, torch.LongTensor(roads), torch.FloatTensor(candidates), torch.LongTensor(candidates_id), trace_lens

# def padding(batch):
#     trace_lens = [len(sample[0]) for sample in batch]
#     # road_lens = [len(sample[1]) for sample in batch]
#     candidates_lens = [len(candidates) for sample in batch for candidates in sample[2]]
#     max_tlen, max_clen = max(trace_lens), max(candidates_lens)
#     traces, roads, candidates, candidates_id = [], [], [], []
#     # 0: [PAD]
#     for sample in batch:
#         traces.append(sample[0] + [[0, 0]] * (max_tlen - len(sample[0])))
#         roads.append(sample[1] + [-1] * (max_tlen - len(sample[1])))
#         candidates.append([candidates + [[0, 0]] * (max_clen-len(candidates)) for candidates in sample[2]] + [[[0, 0]] * max_clen] * (max_tlen - len(sample[2])))
#         candidates_id.append([candidates_id + [-1] * (max_clen-len(candidates_id)) for candidates_id in sample[3]] + [[-1] * max_clen] * (max_tlen - len(sample[3])))
#     return torch.FloatTensor(traces), torch.LongTensor(roads), torch.FloatTensor(candidates), torch.LongTensor(candidates_id), trace_lens