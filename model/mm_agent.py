from memory import Memory
from model.Q_network import QNetwork

import torch
import torch.nn as nn


class MMAgent(nn.Module):
    def __init__(self, correct_reward, mask_reward):
        super(MMAgent, self).__init__()
        self.correct_reward = correct_reward
        self.mask_reward = mask_reward
        self.main_net = QNetwork()
        self.target_net = QNetwork().eval()
        self.memory = Memory(100)

    def select_action(self, last_traces_encoding, last_matched_road_segments_encoding, traces, matched_road_segments_id,
                      candidates, road_graph, trace_graph):
        with torch.no_grad():
            # 使用main_net进行前向传播，得到每个样本的动作Q值
            traces_encoding, matched_road_segments_encoding, action_values = self.main_net(last_traces_encoding,
                                                                                           last_matched_road_segments_encoding,
                                                                                           traces,
                                                                                           matched_road_segments_id,
                                                                                           candidates, road_graph,
                                                                                           trace_graph)
            # 对每个样本选择最大Q值的动作
            return traces_encoding, matched_road_segments_encoding, torch.argmax(action_values, dim=-1)

    def step(self, last_matched_road, road_graph, action, candidates_id, tgt_roads, trace_lens, point_idx):
        seq_len = candidates_id.size(1)
        rewards = [[] for _ in range(seq_len)]  # 计算奖励的张量
        for i in range(action.size(0)):  # 首先遍历批次中的每个样本
            # last_road_id = last_matched_road[i].item()
            for j in range(seq_len):  # 然后遍历样本的序列长度
                # 如果当前点的索引大于轨迹长度，则奖励为0
                if trace_lens[i] <= point_idx + 1 - (seq_len - 1) + j:
                    reward = self.mask_reward
                else:
                    # 从candidates_id中选择由action指定的候选路段ID
                    selected_candidate_id = candidates_id[i, j, action[i, j]]
                    # 根据选中的路段与目标路段是否相同给予奖励
                    reward = self.correct_reward if selected_candidate_id == tgt_roads[i, j] else -1

                    # 根据选中的路段与目标路段距离给予奖励
                    # reward = -road_graph.real_distances.get((selected_candidate_id.item(), tgt_roads[i, j].item()))
                    # if reward == -1:
                    #     connectivity = road_graph.connectivity_distances.get((last_road_id, selected_candidate_id.item()),-1)
                    #     if connectivity == -1:
                    #         reward = reward * 2
                    #     elif connectivity > 2:
                    #         reward = reward * 1.5
                    # reward = min(reward, -1)
                    # else:
                    #     reward = self.correct_reward
                    # last_road_id = selected_candidate_id.item()
                rewards[j].append(reward)

        # 将奖励列表转换为张量
        return torch.tensor(rewards).transpose(0, 1)

    def update_target_net(self):
        # 更新目标网络参数，可以根据需要调整更新频率
        self.target_net.load_state_dict(self.main_net.state_dict())
