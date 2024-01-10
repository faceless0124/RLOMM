import random
from memory import Memory
from model.road_gin import RoadGIN

import torch
import math

import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, road_emb_dim=32, trace_dim=3, traces_d_model=32, candidate_d_model=32, num_layers=3):
        super(QNetwork, self).__init__()
        self.trace_feat_fc = nn.Linear(trace_dim, traces_d_model).cuda()

        # RNN for Traces
        self.rnn_traces = nn.RNN(input_size=traces_d_model, hidden_size=traces_d_model, num_layers=num_layers,
                                 batch_first=True).cuda()

        self.road_feat_fc = nn.Linear(28, road_emb_dim).cuda()  # 3*8 + 4
        self.road_gin = RoadGIN(road_emb_dim)

        # RNN for Road Segments
        self.rnn_segments = nn.RNN(input_size=road_emb_dim, hidden_size=road_emb_dim, num_layers=num_layers,
                                   batch_first=True).cuda()

        self.trace_weight = nn.Linear(traces_d_model, road_emb_dim // 2).cuda()
        self.segment_weight = nn.Linear(road_emb_dim, road_emb_dim // 2).cuda()

        # attention
        self.attention = Attention(32, candidate_d_model, 16)

    def forward(self, traces_encoding, matched_road_segments_encoding, traces, matched_road_segments_id, candidates,
                road_graph):
        # 编码路网特征
        road_x = self.road_feat_fc(road_graph.road_x)
        road_emb = F.relu(road_x)  # 添加ReLU激活函数
        road_emb = self.road_gin(road_emb, road_graph.road_adj)
        # 提取对应路段编码
        segments_emb = road_emb[matched_road_segments_id.squeeze(-1)]
        candidates_emb = road_emb[candidates.squeeze(-1)]
        # 编码轨迹特征
        traces = self.trace_feat_fc(traces)
        traces = F.relu(traces)  # 添加ReLU激活函数
        # 应用RNN模型
        traces_output, traces_hidden = self.rnn_traces(traces, traces_encoding)
        segments_output, segments_hidden = self.rnn_segments(segments_emb, matched_road_segments_encoding)
        # 为轨迹和路段编码添加权重
        traces_encoded = self.trace_weight(traces_output)
        segments_encoded = self.segment_weight(segments_output)
        # 应用注意力机制
        action_values = self.attention(traces_encoded, segments_encoded, candidates_emb)
        # 返回动作值和最终隐藏状态
        return traces_hidden, segments_hidden, action_values


class Attention(nn.Module):
    def __init__(self, combined_dim, candidate_dim, d_model):
        super(Attention, self).__init__()
        self.proj = nn.Linear(combined_dim, d_model)
        self.proj_candidates = nn.Linear(candidate_dim, d_model)

    def forward(self, trace_encoded, segments_encoded, candidates):
        # 将轨迹和路段编码合并
        trace_segments_combined = torch.cat((trace_encoded, segments_encoded), dim=2) # (batch_size, seq_len, combined_dim)
        x_proj = torch.tanh(self.proj(trace_segments_combined))  # (batch_size, seq_len, d_model)
        candidates_proj = torch.tanh(self.proj_candidates(candidates))  # (batch_size, seq_len, candidates_num, d_model)
        x_proj = x_proj.unsqueeze(2)  # (batch_size, seq_len, 1, d_model)
        scores = torch.matmul(x_proj, candidates_proj.transpose(2, 3)).squeeze(2)  # (batch_size, seq_len, num_candidates)
        return scores


class DQNAgent(nn.Module):
    def __init__(self, correct_reward, wrong_reward):
        super(DQNAgent, self).__init__()
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.main_net = QNetwork()
        self.target_net = QNetwork().eval()
        self.memory = Memory(100)

    def select_action(self, last_traces_encoding, last_matched_road_segments_encoding, traces, matched_road_segments_id,
                      candidates, road_graph):
        with torch.no_grad():
            # 使用main_net进行前向传播，得到每个样本的动作Q值
            traces_encoding, matched_road_segments_encoding, action_values = self.main_net(last_traces_encoding,
                                                                                           last_matched_road_segments_encoding,
                                                                                           traces,
                                                                                           matched_road_segments_id,
                                                                                           candidates, road_graph)
            # 对每个样本选择最大Q值的动作
            return traces_encoding, matched_road_segments_encoding, torch.argmax(action_values, dim=-1)

    def step(self, action, candidates_id, tgt_roads, trace_lens, point_idx):
        seq_len = candidates_id.size(1)
        rewards = [[] for _ in range(seq_len)]  # 计算奖励的张量
        for i in range(action.size(0)):  # 首先遍历批次中的每个样本
            for j in range(seq_len):  # 然后遍历样本的序列长度
                # 如果当前点的索引大于轨迹长度，则奖励为0
                if trace_lens[i] <= point_idx + 1 - (seq_len - 1) + j:
                    reward = 0
                else:
                    # 从candidates_id中选择由action指定的候选路段ID
                    selected_candidate_id = candidates_id[i, j, action[i, j]]
                    # 如果选中的路段ID与目标路段ID匹配，则奖励为1，否则为-1
                    reward = self.correct_reward if selected_candidate_id == tgt_roads[i, j] else self.wrong_reward
                rewards[j].append(reward)

        # 将奖励列表转换为张量
        return torch.tensor(rewards).transpose(0, 1)

    def update_target_net(self):
        # 更新目标网络参数，可以根据需要调整更新频率
        self.target_net.load_state_dict(self.main_net.state_dict())
