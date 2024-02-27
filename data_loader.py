import json
import os.path as osp
import pickle
from torch.utils.data import Dataset
import torch
import data_preprocess.utils as utils
import time
import numpy as np


class MyDataset(Dataset):
    def __init__(self, path, name):
        if not path.endswith('/'):
            path += '/'
        self.data_path = osp.join(path, f"{name}_data/{name}.json")
        self.MIN_LAT, self.MIN_LNG, self.MAX_LAT, self.MAX_LNG = utils.get_border('./data/road.txt')
        self.map_path = osp.join(path, "used_pkl/grid2traceid_dict.pkl")
        self.buildingDataset(self.data_path)

    def buildingDataset(self, data_path):
        grid2traceid_dict = pickle.load(open(self.map_path, 'rb'))
        self.traces_ls = []

        with open(data_path, "r") as fp:
            data = json.load(fp)
            # self.traces_ls = data[0::5]
            for gps_ls in data[0::5]:
                traces = []
                for gps in gps_ls:
                    gridx, gridy = utils.gps2grid(gps[0], gps[1], MIN_LAT=self.MIN_LAT, MIN_LNG=self.MIN_LNG)
                    traces.append(grid2traceid_dict[(gridx, gridy)] + 1)
                self.traces_ls.append(traces)
            self.time_stamps = data[1::5]
            self.roads_ls = data[2::5]
            self.candidates = data[3::5]
            self.candidates_id = data[4::5]
        self.length = len(self.traces_ls)

        # 归一化轨迹和候选点数据
        # self.traces_ls = [self.normalize_traces(trace) for trace in self.traces_ls]
        # self.traces_ls = [self.trace_gps2grid(trace) for trace in self.traces_ls]
        # self.candidates = [self.normalize_candidates(candidate) for candidate in self.candidates]
        self.time_stamps = [self.normalize_time_stamps(ts) for ts in self.time_stamps]

        self.cnt = self.count_point()

        print(self.data_path)
        print("Total points: ", self.cnt)
        print(len(self.traces_ls), len(self.time_stamps), len(self.roads_ls), len(self.candidates),
              len(self.candidates_id))
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

    def trace_gps2grid(self, trace):
        # 经纬度转网格
        return [list(utils.gps2grid(lat, lng, MIN_LAT=self.MIN_LAT, MIN_LNG=self.MIN_LNG))
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

    # def normalize_time_stamps(self, time_stamps):
    #     # 将第一个时间点转换为秒
    #     base_time = time.mktime(time.strptime(time_stamps[0], '%Y/%m/%d %H:%M:%S'))
    #     # 计算每个时间点与第一个时间点的时间间隔
    #     normalized_time_stamps = [time.mktime(time.strptime(ts, '%Y/%m/%d %H:%M:%S')) - base_time for ts in time_stamps]
    #     return normalized_time_stamps

    # def normalize_time_stamps(self, time_stamps):
    #     # 初始化一个空列表来存储计算出的时间间隔
    #     normalized_time_stamps = []
    #
    #     # 遍历时间戳列表，计算每个时间点与上一个时间点的时间间隔
    #     for i, ts in enumerate(time_stamps):
    #         if i == 0:
    #             # 第一个时间点的时间间隔设为0
    #             normalized_time_stamps.append(0)
    #         else:
    #             # 计算当前时间点与前一个时间点的时间间隔
    #             current_time = time.mktime(time.strptime(ts, '%Y/%m/%d %H:%M:%S'))
    #             previous_time = time.mktime(time.strptime(time_stamps[i - 1], '%Y/%m/%d %H:%M:%S'))
    #             interval = current_time - previous_time
    #             normalized_time_stamps.append(interval)
    #
    #     return normalized_time_stamps

    def normalize_time_stamps(self, time_stamps):
        # 初始化一个空列表来存储计算出的时间特征
        normalized_time_features = []

        # 初始化前一个时间戳的时间（秒）
        previous_seconds = None

        # 定义一个函数来判断时间段
        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return 0  # 早晨
            elif 12 <= hour < 17:
                return 1  # 上午
            elif 17 <= hour < 21:
                return 2  # 下午
            else:
                return 3  # 晚上

        # 遍历时间戳列表
        for ts in time_stamps:
            # 解析当前时间戳
            current_time = time.strptime(ts, '%Y/%m/%d %H:%M:%S')
            # 计算当前时间戳对应的总秒数
            current_seconds = current_time.tm_hour * 3600 + current_time.tm_min * 60 + current_time.tm_sec
            # 如果是第一个时间戳，则时间差设为0
            if previous_seconds is None:
                time_diff = 0
            else:
                # 计算与前一时间戳的时间差（秒）
                time_diff = current_seconds - previous_seconds
            # 更新前一个时间戳的时间
            previous_seconds = current_seconds
            # 提取一天中的时间（分钟数）
            minutes_since_midnight = current_time.tm_hour * 60 + current_time.tm_min
            # 提取一周中的天数
            day_of_week = current_time.tm_wday
            # 获取时间段
            time_of_day = get_time_of_day(current_time.tm_hour)
            # 将这些特征合并为一个特征向量
            features = [minutes_since_midnight, day_of_week, time_diff, time_of_day]
            # 将特征向量添加到结果列表中
            normalized_time_features.append(features)

        return np.array(normalized_time_features)

    def __getitem__(self, index):
        return self.traces_ls[index], self.time_stamps[index], self.roads_ls[index], \
            self.candidates[index], self.candidates_id[index]

    def __len__(self):
        return self.length


def padding(batch):
    trace_lens = [len(sample[0]) for sample in batch]
    # road_lens = [len(sample[1]) for sample in batch]
    candidates_lens = [len(candidates) for sample in batch for candidates in sample[3]]
    max_tlen, max_clen = max(trace_lens), max(candidates_lens)
    traces, time_stamp, roads, candidates, candidates_id = [], [], [], [], []
    # 0: [PAD]
    for sample in batch:
        # traces.append(sample[0] + [[0, 0]] * (max_tlen - len(sample[0])))
        traces.append(sample[0] + [0] * (max_tlen - len(sample[0])))

        time_stamp_sample = sample[1]
        if len(time_stamp_sample) != max_tlen:
            padding = np.array([[-1] * 4] * (max_tlen - len(time_stamp_sample)))  # 创建填充部分
            time_stamp_sample = np.concatenate((time_stamp_sample, padding), axis=0)
        time_stamp.append(time_stamp_sample)

        # time_stamp.append(sample[1] + [-1] * (max_tlen - len(sample[1])))
        roads.append(sample[2] + [-1] * (max_tlen - len(sample[2])))
        candidates.append([candidates + [0] * (max_clen - len(candidates)) for candidates in sample[3]] + [
            [0] * max_clen] * (max_tlen - len(sample[3])))
        # candidates.append([candidates + [[0,0]] * (max_clen - len(candidates)) for candidates in sample[3]] + [
        #     [[0,0]] * max_clen] * (max_tlen - len(sample[3])))
        candidates_id.append(
            [candidates_id + [8533] * (max_clen - len(candidates_id)) for candidates_id in sample[4]] + [
                [8533] * max_clen] * (max_tlen - len(sample[4])))

    traces_array = np.array(traces)  # 将traces列表转换为NumPy数组
    time_stamp_array = np.array(time_stamp)  # 将time_stamp列表转换为NumPy数组
    traces_tensor = torch.FloatTensor(traces_array).unsqueeze(-1)  # 转换为张量并增加一个维度
    time_stamp_tensor = torch.FloatTensor(time_stamp_array)  # 转换为张量
    traces = torch.cat((traces_tensor, time_stamp_tensor), dim=-1)
    return traces, torch.LongTensor(roads), torch.FloatTensor(candidates), torch.LongTensor(candidates_id), trace_lens
