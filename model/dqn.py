import random
from memory import Memory
from model.road_gin import RoadGIN


import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class QNetwork(nn.Module):
    def __init__(self, action_dim, road_emb_dim=16, traces_d_model=2, candidate_d_model=2, nhead=2, num_layers=2):
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

    def forward(self, traces, matched_road_segments_id, candidates, road_graph):

        road_x = self.road_feat_fc(road_graph.road_x)
        road_emb = self.road_gin(road_x, road_graph.road_adj)

        # Calculate the dimensions of traces and segments
        if matched_road_segments_id.size(0) == 1:
            segments_emb = torch.zeros(1, self.road_emb_dim).to(traces.device)
        else:
            print(matched_road_segments_id[1:].shape)
            segments_emb = road_emb[matched_road_segments_id[1:]].squeeze(0)

        print(segments_emb.shape)
        trace_len = traces.size(0)
        segment_len = segments_emb.size(0)

        print("trace_len:", trace_len, "segment_len:", segment_len)

        # print("trace_len:", trace_len, "segment_len:", segment_len)

        # print("trace:", traces.shape, "segments_emb:", segments_emb.shape)

        # Positional Encoding for Traces and Road Segments
        # Initialize the positional encoding matrix
        pe_trace = self.positional_encoding(trace_len, traces.size(-1)).to(traces.device)
        traces = traces + pe_trace

        pe_segments = self.positional_encoding(segment_len, segments_emb.size(-1)).to(segments_emb.device)
        segments_emb = segments_emb + pe_segments

        # Permute to fit transformer input format
        traces = traces.unsqueeze(0)
        traces = traces.permute(1, 0, 2)
        segments_emb = segments_emb.unsqueeze(0)
        segments_emb = segments_emb.permute(1, 0, 2)

        # print("trace:", traces.shape, "segments_emb:", segments_emb.shape)

        # Apply Transformer Encoder for Traces
        traces_encoded = self.transformer_encoder_traces(traces)
        traces_encoded = traces_encoded[-1]  # Take the last across sequence dimension

        # Apply Transformer Encoder for Road Segments
        segments_encoded = self.transformer_encoder_segments(segments_emb)
        segments_encoded = segments_encoded.mean(dim=0)  # Take the mean across sequence dimension

        # attention
        traces_encoded_expanded = traces_encoded.repeat(candidates.size(0), 1)
        segments_encoded_expanded = segments_encoded.repeat(candidates.size(0), 1)
        combined_features = torch.cat((traces_encoded_expanded, segments_encoded_expanded, candidates), dim=1)

        # 应用注意力机制
        attention_output, _ = self.attention(combined_features)

        # 计算 Q 值
        action_values = attention_output.squeeze(-1)

        # print("candidates:", candidates)
        print("candidates:", candidates)
        print("action_values:", action_values)

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
        return weighted.sum(1), attention_weights


class DQNAgent(nn.Module):
    def __init__(self, action_dim):
        super(DQNAgent, self).__init__()
        self.action_dim = action_dim
        self.policy_net = QNetwork(action_dim)
        self.target_net = QNetwork(action_dim).eval()
        self.memory = Memory(10000)

    def select_action(self, state, matched_road_segments_id, road_graph, candidates, steps_done, EPS_START=0.9, EPS_END=0.1, EPS_DECAY=2000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        if sample > 0:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state, matched_road_segments_id, candidates, road_graph))
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)

    def step(self,action, candidates_id, tgt_roads):
        # return reward
        if candidates_id[action] == tgt_roads:
            return 1
        else:
            return -1

    def update_target_net(self):
        # 更新目标网络参数，可以根据需要调整更新频率
        self.target_net.load_state_dict(self.policy_net.state_dict())

