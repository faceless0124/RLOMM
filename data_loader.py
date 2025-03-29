import json
import os.path as osp
import pickle
from torch.utils.data import Dataset
import torch
import data_preprocess.utils as utils
import time
import numpy as np


class MyDataset(Dataset):
    def __init__(self, path, name, city):
        if not path.endswith('/'):
            path += '/'
        self.data_path = osp.join(path, f"{name}_data/{name}.json")
        self.MIN_LAT, self.MIN_LNG, self.MAX_LAT, self.MAX_LNG = utils.get_border('data/' + city + '_road.txt')
        self.map_path = osp.join(path, "used_pkl/grid2traceid_dict.pkl")
        self.buildingDataset(self.data_path)
        if city == 'beijing':
            self.link_cnt = 8533
        elif city == 'porto':
            self.link_cnt = 4254
        else:
            self.link_cnt = 6576

    def buildingDataset(self, data_path, subset_ratio=0.8):
        grid2traceid_dict = pickle.load(open(self.map_path, 'rb'))
        self.traces_ls = []

        with open(data_path, "r") as fp:
            data = json.load(fp)
            total_data_points = len(data[0::4])
            subset_size = int(total_data_points * subset_ratio)

            # Adjust subset size to ensure it's within the range of available data
            subset_size = min(subset_size, total_data_points)

            # Select subset of data indices
            subset_indices = range(0, total_data_points, total_data_points // subset_size)[:subset_size]

            for idx in subset_indices:
                gps_ls = data[0::4][idx]
                traces = []
                for gps in gps_ls:
                    gridx, gridy = utils.gps2grid(gps[0], gps[1], MIN_LAT=self.MIN_LAT, MIN_LNG=self.MIN_LNG)
                    traces.append(grid2traceid_dict[(gridx, gridy)] + 1)
                self.traces_ls.append(traces)

            self.time_stamps = [data[1::4][idx] for idx in subset_indices]
            self.tgt_roads_ls = [data[2::4][idx] for idx in subset_indices]
            self.candidates_id = [data[3::4][idx] for idx in subset_indices]

        self.length = len(self.traces_ls)
        self.time_stamps = [self.normalize_time_stamps(ts) for ts in self.time_stamps]

        self.cnt = self.count_point()

        print(self.data_path)
        print("Total points: ", self.cnt)
        print(len(self.traces_ls), len(self.time_stamps), len(self.tgt_roads_ls), len(self.candidates_id))
        assert len(self.traces_ls) == len(self.tgt_roads_ls)
        assert len(self.traces_ls) == len(self.candidates_id)
        assert len(self.traces_ls) == len(self.time_stamps)

    def count_point(self):
        count = 0
        for trace in self.traces_ls:
            for point in trace:
                count += 1
        return count

    def normalize_traces(self, trace):
        return [[(lat - self.MIN_LAT) / (self.MAX_LAT - self.MIN_LAT),
                 (lng - self.MIN_LNG) / (self.MAX_LNG - self.MIN_LNG)]
                for lat, lng in trace]

    def trace_gps2grid(self, trace):
        return [list(utils.gps2grid(lat, lng, MIN_LAT=self.MIN_LAT, MIN_LNG=self.MIN_LNG))
                for lat, lng in trace]

    def normalize_candidates(self, candidates):
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
        normalized_time_stamps = [0]

        for i in range(1, len(time_stamps)):
            current_time = time.mktime(time.strptime(time_stamps[i], '%Y/%m/%d %H:%M:%S'))
            previous_time = time.mktime(time.strptime(time_stamps[i - 1], '%Y/%m/%d %H:%M:%S'))

            interval = current_time - previous_time
            normalized_time_stamps.append(interval)

        return normalized_time_stamps

    def __getitem__(self, index):
        return self.traces_ls[index], self.time_stamps[index], self.tgt_roads_ls[index], self.candidates_id[index]

    def __len__(self):
        return self.length


    def padding(self, batch):
        trace_lens = [len(sample[0]) for sample in batch]
        candidates_lens = [len(candidates) for sample in batch for candidates in sample[3]]
        max_tlen, max_clen = max(trace_lens), max(candidates_lens)
        traces, time_stamp, tgt_roads, candidates_id = [], [], [], []
        # 0: [PAD]
        for sample in batch:
            traces.append(sample[0] + [0] * (max_tlen - len(sample[0])))

            time_stamp.append(sample[1] + [-1] * (max_tlen - len(sample[1])))
            tgt_roads.append(sample[2] + [0] * (max_tlen - len(sample[2])))

            candidates_id.append(
                [candidates_id + [self.link_cnt] * (max_clen - len(candidates_id)) for candidates_id in sample[3]] + [
                    [self.link_cnt] * max_clen] * (max_tlen - len(sample[3])))

        traces_array = np.array(traces)
        time_stamp_array = np.array(time_stamp)
        traces_tensor = torch.FloatTensor(traces_array).unsqueeze(-1)
        time_stamp_tensor = torch.FloatTensor(time_stamp_array).unsqueeze(-1)
        traces = torch.cat((traces_tensor, time_stamp_tensor), dim=-1)
        return traces, torch.LongTensor(tgt_roads), torch.LongTensor(candidates_id), trace_lens
