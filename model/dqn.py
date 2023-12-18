import random
from memory import Memory
from model.road_gin import RoadGIN


import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class QNetwork(nn.Module):
    def __init__(self, road_emb_dim=16, traces_d_model=2, candidate_d_model=2, nhead=2, num_layers=2):
        super(QNetwork, self).__init__()
        self.road_emb_dim = road_emb_dim

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
        self.attention = Attention(traces_d_model + road_emb_dim + candidate_d_model, None)

    def forward(self, traces, matched_road_segments_id, candidates, road_graph, seq_len, trace_len_ls=None):
        # 获取路网特征
        road_x = self.road_feat_fc(road_graph.road_x)
        road_emb = self.road_gin(road_x, road_graph.road_adj)

        # max_seq_len = max(tensor.shape[0] for tensor in sub_traces)
        # # 对每个 tensor 进行填充
        # padded_traces = []
        # for tensor in sub_traces:
        #     # 计算需要填充的长度
        #     pad_len = max_seq_len - tensor.shape[0]
        #
        #     # 如果需要填充，则进行填充
        #     if pad_len > 0:
        #         # 在 seq_len 维度上填充 (0,0) 对
        #         pad_tensor = F.pad(tensor, (0, 0, 0, pad_len), 'constant', 0)
        #         padded_traces.append(pad_tensor)
        #     else:
        #         padded_traces.append(tensor)

        # sub_traces = torch.stack(sub_traces, dim=0)  # 将列表转换为张量

        segments_emb = road_emb[matched_road_segments_id.squeeze(-1)]

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
        traces_encoded = traces_encoded.unsqueeze(1).repeat(1, candidates.size(1), 1)
        segments_encoded = segments_encoded.unsqueeze(1).repeat(1, candidates.size(1), 1)

        # 应用注意力机制
        combined_features = torch.cat((traces_encoded, segments_encoded, candidates), dim=2)
        attention_output, _ = self.attention(combined_features)
        action_values = attention_output.squeeze(-1)

        return action_values

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
    def __init__(self, feature_dim, step_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.proj = nn.Linear(feature_dim, 64)
        self.context_vector = nn.Linear(64, 1, bias=False)

    def forward(self, x):
        x_proj = torch.tanh(self.proj(x))
        context_vector = self.context_vector(x_proj).squeeze(-1)
        attention_weights = F.softmax(context_vector, dim=-1)
        weighted = torch.mul(x, attention_weights.unsqueeze(-1).expand_as(x))
        return weighted.sum(2), attention_weights


class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.policy_net = QNetwork()
        self.target_net = QNetwork().eval()
        self.memory = Memory(10000)

    def select_action(self, traces, matched_road_segments_id, candidates, road_graph, trace_len, seq_len, steps_done=None, evaluate=True, EPS_START=0.9, EPS_END=0.1,
                      EPS_DECAY=2000):
        with torch.no_grad():
            # 使用policy_net进行前向传播，得到每个样本的动作Q值
            action_values = self.policy_net(traces, matched_road_segments_id, candidates, road_graph, seq_len)
            # 对每个样本选择最大Q值的动作
            return torch.argmax(action_values, dim=1)

        # traces, matched_road_segments_id, candidates = state
        # sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        # if sample > eps_threshold or evaluate:
        #     with torch.no_grad():
        #         return torch.argmax(self.policy_net(traces, matched_road_segments_id, candidates, road_graph))
        # else:
        #     return torch.tensor(random.randrange(candidates.size(0)), dtype=torch.long).to(traces.device)

    def step(self, action, candidates_id, tgt_roads):
        # 计算奖励的张量
        rewards = []
        for i in range(action.size(0)):  # 遍历批次中的每个样本
            # 从candidates_id中选择由action指定的候选路段ID
            selected_candidate_id = candidates_id[i, action[i]]
            # 如果选中的路段ID与目标路段ID匹配，则奖励为1，否则为-1
            reward = 1 if selected_candidate_id == tgt_roads[i] else -1
            rewards.append(reward)
        # 将奖励列表转换为张量
        return torch.tensor(rewards)

    def update_target_net(self):
        # 更新目标网络参数，可以根据需要调整更新频率
        self.target_net.load_state_dict(self.policy_net.state_dict())

