from model.road_gin import RoadGIN
from model.trace_gcn import TraceGCN
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, road_emb_dim=128, traces_emb_dim=128, candidate_d_model=128, num_layers=3):
        super(QNetwork, self).__init__()
        self.emb_dim = traces_emb_dim
        self.traces_linear = nn.Linear(2 * self.emb_dim, self.emb_dim).cuda()
        # RNN for Traces
        self.rnn_traces = nn.RNN(input_size=traces_emb_dim, hidden_size=traces_emb_dim, num_layers=num_layers,
                                 batch_first=True).cuda()

        self.road_feat_fc = nn.Linear(28, road_emb_dim).cuda()  # 3*8 + 4
        self.trace_feat_fc = nn.Linear(4, traces_emb_dim).cuda()

        self.road_gin = RoadGIN(road_emb_dim)
        self.trace_gcn = TraceGCN(traces_emb_dim)

        # RNN for Road Segments
        self.rnn_segments = nn.RNN(input_size=road_emb_dim, hidden_size=road_emb_dim, num_layers=num_layers,
                                   batch_first=True).cuda()

        self.trace_weight = nn.Linear(traces_emb_dim, traces_emb_dim//2).cuda()
        self.segment_weight = nn.Linear(road_emb_dim, road_emb_dim//2).cuda()

        # attention
        self.attention = Attention(traces_emb_dim, candidate_d_model, traces_emb_dim)

    def forward(self, traces_encoding, matched_road_segments_encoding, traces, matched_road_segments_id, candidates,
                road_graph, trace_graph):
        # 编码路网特征
        road_emb = self.road_feat_fc(road_graph.road_x)
        road_emb = self.road_gin(road_emb, road_graph.road_adj)
        # 提取对应路段编码
        segments_emb = road_emb[matched_road_segments_id.squeeze(-1)]
        candidates_emb = road_emb[candidates.squeeze(-1)]
        # 编码轨迹特征
        # trace_graph.map_matrix [10551, 8533]
        # road_emb [8534, 32]
        pure_grid_feat = torch.mm(trace_graph.map_matrix, road_emb[:-1, :])
        pure_grid_feat[trace_graph.singleton_grid_mask] = self.trace_feat_fc(trace_graph.singleton_grid_location)
        full_grid_emb = torch.zeros(trace_graph.num_grids + 1, 2 * self.emb_dim).to(traces.device)
        full_grid_emb[1:, :] = self.trace_gcn(pure_grid_feat,
                                              trace_graph.trace_in_edge_index,
                                              trace_graph.trace_out_edge_index,
                                              trace_graph.trace_weight)
        # 提取对应路段编码
        traces_emb = full_grid_emb[traces]
        traces_emb = self.traces_linear(traces_emb)
        # 应用RNN模型
        traces_output, traces_hidden = self.rnn_traces(traces_emb, traces_encoding)
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