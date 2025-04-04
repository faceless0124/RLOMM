import os
import pickle
import random
import json
import math
import sys
import heapq
import re

from tqdm import tqdm

from utils import create_dir, get_border

downsample_rate = sys.argv[1]
city = sys.argv[2]
MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = get_border('../data/' + city + '_road.txt')


def randomDownSampleBySize(sampleData: list, sampleRate: float) -> (list, list, list):
    """
        randomly sampling
    """
    resData, pureData, resIdx = [], [], []
    for i in range(len(sampleData)):
        trajList = sampleData[i]
        tempRes = [trajList[0], trajList[1]]
        tmpIdx = [0]
        for j in range(2, len(trajList) - 1):
            if (random.random() <= sampleRate):
                tempRes.append(trajList[j])
                tmpIdx.append(j - 1)
        tempRes.append(trajList[-1])
        tmpIdx.append(len(trajList) - 2)
        resData.append(tempRes)
        pureData.append(trajList)
        resIdx.append(tmpIdx)
    return resData, pureData, resIdx


class DataProcess():
    def __init__(self, traj_input_path, output_dir,
                 sample_rate, max_road_len=25, min_road_len=15) -> None:
        self.traj_input_path = traj_input_path
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.max_road_len = max_road_len
        self.min_road_len = min_road_len
        beginLs = self.readTrajFile(traj_input_path)
        self.finalLs = self.cutData(beginLs)
        self.traces_ls, self.tgt_roads_ls, self.candidates_id, self.time_stamp, self.downsampleIdx, downSampleData = self.sampling()
        self.splitData(output_dir)
        with open(self.output_dir + 'data_split/downsample_trace.txt', 'w') as f:
            for traces in downSampleData:
                for trace in traces:
                    f.write(trace)

    def readTrajFile(self, filePath):
        """
            read beijing_trace.txt
        """
        with open(filePath, 'r') as f:
            traj_list = f.readlines()
        finalLs = list()
        tempLs = list()
        for idx, sen in enumerate(traj_list):
            if sen[0] == '#':
                if idx != 0:
                    finalLs.append(tempLs)
                tempLs = [sen]
            else:
                tempLs.append(sen)
        finalLs.append(tempLs)
        print("begin traces cnt:", len(finalLs))
        return finalLs

    def closest_point_on_line(self, x, y, a):
        """
        Find the closest point on line segment (x, y) to point a.

        Parameters:
        - x, y: Coordinates of the endpoints of the line segment.
        - a: Coordinates of the point.

        Returns:
        - closest_point: Coordinates of the closest point on the line segment to point a.
        """

        def dot(v1, v2):
            return v1[0] * v2[0] + v1[1] * v2[1]

        def length(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2)

        def subtract(v1, v2):
            return (v1[0] - v2[0], v1[1] - v2[1])

        def scale(v, s):
            return (v[0] * s, v[1] * s)

        line_vector = subtract(y, x)
        point_vector = subtract(a, x)

        line_length = length(line_vector)
        if line_length == 0:
            return x

        line_unit_vector = scale(line_vector, 1.0 / line_length)

        t = dot(point_vector, line_unit_vector)

        if t < 0:
            closest_point = x
        elif t > line_length:
            closest_point = y
        else:
            closest_point = (x[0] + line_unit_vector[0] * t, x[1] + line_unit_vector[1] * t)

        return closest_point

    def get_road_candidates(self, target_link_id: int, road_distance_data: dict) -> (list, list):
        """
        Find candidate points of target_link_id.

        Parameters:
        - path: road info data path.
        - target_link_id: the target link id.

        Returns:
        - candidate_ids: IDs of the 10 closest road segments to target_link_id.
        - candidate_distances: distances of these road segments from target_link_id.
        """
    
        closest_candidates = []
        if city == 'beijing':
            link_cnt = 8533
        elif city == 'porto':
            link_cnt = 4254
        else:
            link_cnt = 6576
        for candidate_id in range(link_cnt):
            distance = road_distance_data.get((target_link_id, candidate_id))
            if distance is not None:
                if len(closest_candidates) < 10:
                    heapq.heappush(closest_candidates, (-distance, candidate_id))
                elif distance < -closest_candidates[0][0]:
                    heapq.heappushpop(closest_candidates, (-distance, candidate_id))

        candidate_ids = [item[1] for item in closest_candidates]
        # candidate_distances = [-item[0] for item in closest_candidates]
        # print(candidate_distances)


        return candidate_ids


    def sampling(self):
        """
            down sampling
        """
        path = '../data/' + city + '/real_distances.pkl'
        downsampleData, pureData, downsampleIdx = randomDownSampleBySize(self.finalLs, self.sample_rate)
        traces_ls, tgt_roads_ls, candidates_id_ls, time_stamp_ls =  [], [], [], []
        with open(path, 'rb') as file:
            road_distance_data = pickle.load(file)
            for downdata in tqdm(downsampleData):
                traces, roads, candidates_ids, time_stamp = [], [], [], []
                for i in downdata:
                    if i[0] == '#':
                        continue
                    il = i.split(',')
                    time_stamp.append(il[0])
                    lat, lng = float(il[1]), float(il[2])
                    # a = (lng, lat)
                    traces.append((lat, lng))
                    candidates_id = self.get_road_candidates(int(i.split(',')[3]), road_distance_data)
                    candidates_ids.append(candidates_id)
                    roads.append(candidates_id.index(int(i.split(',')[3])))

                traces_ls.append(traces)
                tgt_roads_ls.append(roads)
                candidates_id_ls.append(candidates_ids)
                time_stamp_ls.append(time_stamp)
        return traces_ls, tgt_roads_ls, candidates_id_ls, time_stamp_ls, downsampleIdx, downsampleData

    def cutData(self, beginLs):
        """
            ensure each trace's length in [min_lens+1, max_lens+min_lens+1)
        """
        finalLs = []
        for traces in beginLs:
            assert traces[0][0] == '#'
            title = traces[0]
            traces = traces[1:]
            lens = len(traces)
            if lens < self.min_road_len:
                continue
            if lens < self.max_road_len and lens >= self.min_road_len:
                finalLs.append([title] + traces)
            else:
                cutnum = lens / self.max_road_len
                int_cutnum = int(cutnum)
                lstnum = lens - int_cutnum * self.max_road_len
                if lens % self.max_road_len != 0:
                    int_cutnum += 1
                else:
                    lstnum = self.max_road_len
                if lstnum < self.min_road_len:
                    int_cutnum -= 1

                for i in range(int_cutnum - 1):
                    tmp_ls = [title] + traces[i * self.max_road_len:(i + 1) * self.max_road_len]
                    finalLs.append(tmp_ls)

                assert (lens - (int_cutnum - 1) * self.max_road_len < self.max_road_len + self.min_road_len)
                latLS = [title] + traces[(int_cutnum - 1) * self.max_road_len:]

                finalLs.append(latLS)

        for i in finalLs:
            assert (len(i) >= 16 and len(i) <= 40)
        print("traces cnt:", len(finalLs))
        return finalLs

    def splitData(self, output_dir, train_rate=0.7, val_rate=0.2):
        """
            split original data to train, valid and test datasets
        """
        create_dir(output_dir)
        create_dir(output_dir + 'data_split/')
        train_data_dir = output_dir + 'train_data/'
        create_dir(train_data_dir)
        val_data_dir = output_dir + 'val_data/'
        create_dir(val_data_dir)
        test_data_dir = output_dir + 'test_data/'
        create_dir(test_data_dir)
        num_sample = len(self.traces_ls)
        train_size, val_size = int(num_sample * train_rate), int(num_sample * val_rate)
        idxs = list(range(num_sample))
        random.shuffle(idxs)
        train_idxs = idxs[:train_size]
        val_idxs = idxs[train_size:train_size + val_size]
        trainset, valset, testset = [], [], []

        train_trace = []
        val_trace = []
        test_trace = []
        for i in range(num_sample):
            if i in train_idxs:
                trainset.extend([self.traces_ls[i], self.time_stamp[i], self.tgt_roads_ls[i], self.candidates_id[i]])
                train_trace += [self.finalLs[i]]
            elif i in val_idxs:
                valset.extend([self.traces_ls[i], self.time_stamp[i], self.tgt_roads_ls[i], self.candidates_id[i]])
                val_trace += [self.finalLs[i]]
            else:
                testset.extend([self.traces_ls[i], self.time_stamp[i], self.tgt_roads_ls[i], self.candidates_id[i]])
                test_trace += [self.finalLs[i]]

        with open(os.path.join(train_data_dir, "train.json"), 'w') as fp:
            json.dump(trainset, fp)

        with open(os.path.join(val_data_dir, "val.json"), 'w') as fp:
            json.dump(valset, fp)

        with open(os.path.join(test_data_dir, "test.json"), 'w') as fp:
            json.dump(testset, fp)


        all_trace = [train_trace, val_trace, test_trace]
        all_trace_name = ['train_trace.txt', 'val_trace.txt', 'test_trace.txt']
        for i in range(3):
            tmptrace = all_trace[i]
            path = output_dir + 'data_split/' + all_trace_name[i]
            with open(path, 'w') as f:
                for traces in tmptrace:
                    for trace in traces:
                        f.write(trace)


if __name__ == "__main__":
    path = '../data/' + city + '/'
    data_path = path + 'data' + downsample_rate + '_0.1' + '/'
    DataProcess(traj_input_path=path + city + '_trace.txt', output_dir=data_path, sample_rate=float(downsample_rate))
