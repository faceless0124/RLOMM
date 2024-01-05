import random
from memory import Memory
from model.road_gin import RoadGIN

import torch.nn as nn
import torch
import math


class QNetwork(nn.Module):
    def __init__(self, road_emb_dim=16, trace_dim=3, traces_d_model=16, candidate_d_model=16, nhead=1, num_layers=3):
        super(QNetwork, self).__init__()
        self.road_emb_dim = road_emb_dim

        self.trace_feat_fc = nn.Linear(trace_dim, traces_d_model).cuda()
        # Transformer Encoder for Traces
        self.transformer_encoder_traces = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=traces_d_model, nhead=nhead),
            num_layers=num_layers
        )

        self.road_feat_fc = nn.Linear(28, road_emb_dim).cuda()  # 3*8 + 4
        self.road_gin = RoadGIN(road_emb_dim)

        # Transformer Encoder for Road Segments
        self.transformer_encoder_segments = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=road_emb_dim, nhead=nhead),
            num_layers=num_layers
        )

        # attention
        self.attention = Attention(traces_d_model + road_emb_dim, candidate_d_model, 32)

    def forward(self, traces, matched_road_segments_id, candidates, road_graph, seq_len, trace_len_ls=None):
        # 获取路网特征
        road_x = self.road_feat_fc(road_graph.road_x)
        road_emb = self.road_gin(road_x, road_graph.road_adj)

        segments_emb = road_emb[matched_road_segments_id.squeeze(-1)]
        candidates_emb = road_emb[candidates.squeeze(-1)]

        traces = self.trace_feat_fc(traces)

        # 应用位置编码和变换器编码器
        pe_trace = self.positional_encoding(traces.size(1), traces.size(-1)).to(traces.device)
        traces = traces + pe_trace.unsqueeze(0)
        # 类似地处理segments_emb
        pe_segments = self.positional_encoding(segments_emb.size(1), segments_emb.size(-1)).to(segments_emb.device)
        segments_emb = segments_emb + pe_segments.unsqueeze(0)

        # 对批处理数据应用变换器编码器
        traces_encoded = self.transformer_encoder_traces(traces)
        traces_encoded = traces_encoded[:, -1, :]  # 取最后一个元素
        segments_encoded = self.transformer_encoder_segments(segments_emb)
        segments_encoded = segments_encoded.mean(dim=1)  # 对每个序列取平均

        # 如果candidates包含变长序列，您可能需要进行相应的处理
        traces_encoded = traces_encoded.unsqueeze(1) # (batch_size, 1, traces_d_model)
        segments_encoded = segments_encoded.unsqueeze(1) # (batch_size, 1, road_emb_dim)
        # candidates.shape = (batch_size, num_candidates, candidate_d_model)
        # 应用注意力机制
        action_values = self.attention(traces_encoded, segments_encoded, candidates_emb)

        return action_values
        # output_tensor = torch.zeros(candidates.size(0), candidates.size(1)).to(candidates.device)
        #
        # # 将每个batch的第一个元素设置为1
        # output_tensor[:, 0] = 1
        # return output_tensor

    def positional_encoding(self, seq_len, d_model):
        """
        Generate positional encoding for a given sequence length and model dimension.

        Parameters:
        seq_len (int): Length of the sequence or time steps in the sequence.
        d_model (int): The dimension of the embeddings or the feature dimension.

        Returns:
        torch.Tensor: A tensor of shape [seq_len, d_model] containing positional encodings.
        """
        # Initialize the positional encoding matrix
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # Compute the positional encodings
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def get_road_emb(self, gdata):
        """
        get road embedding embedding
        """
        # road embedding, (num_roads, embed_dim)
        road_x = self.road_feat_fc(gdata.road_x)
        full_road_emb = self.road_gin(road_x, gdata.road_adj)

        return full_road_emb


class Attention(nn.Module):
    def __init__(self, combined_dim, candidate_dim, d_model):
        super(Attention, self).__init__()
        self.proj = nn.Linear(combined_dim, d_model)
        self.proj_candidates = nn.Linear(candidate_dim, d_model)

    def forward(self, trace_encoded, segments_encoded, candidates):
        # 将轨迹和路段编码合并
        trace_segments_combined = torch.cat((trace_encoded, segments_encoded), dim=-1)  # (batch_size, 1, combined_dim)
        x_proj = torch.tanh(self.proj(trace_segments_combined))  # (batch_size, 1, d_model)
        candidates_proj = torch.tanh(self.proj_candidates(candidates))  # (batch_size, num_candidates, d_model)
        # Compute attention scores between each x and the candidates
        scores = torch.matmul(x_proj, candidates_proj.transpose(1, 2)).squeeze(1)  # (batch_size, 1, num_candidates)
        return scores


class DQNAgent(nn.Module):
    def __init__(self, correct_reward, wrong_reward):
        super(DQNAgent, self).__init__()
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.main_net = QNetwork()
        self.target_net = QNetwork().eval()
        self.memory = Memory(100)

    def select_action(self, traces, matched_road_segments_id, candidates, road_graph, seq_len):
        with torch.no_grad():
            # 使用main_net进行前向传播，得到每个样本的动作Q值
            action_values = self.main_net(traces, matched_road_segments_id, candidates, road_graph, seq_len)
            # 对每个样本选择最大Q值的动作
            return torch.argmax(action_values, dim=1)

    def step(self, action, candidates_id, tgt_roads, trace_lens, point_idx):
        # 计算奖励的张量
        rewards = []
        for i in range(action.size(0)):  # 遍历批次中的每个样本
            # 如果当前点的索引大于轨迹长度，则奖励为0
            if trace_lens[i] <= point_idx + 1:
                reward = 0
            else:
                # 从candidates_id中选择由action指定的候选路段ID
                selected_candidate_id = candidates_id[i, action[i]]
                # 如果选中的路段ID与目标路段ID匹配，则奖励为1，否则为-1
                reward = self.correct_reward if selected_candidate_id == tgt_roads[i] else self.wrong_reward

            rewards.append(reward)
        # 将奖励列表转换为张量
        return torch.tensor(rewards)

    def update_target_net(self):
        # 更新目标网络参数，可以根据需要调整更新频率
        self.target_net.load_state_dict(self.main_net.state_dict())
