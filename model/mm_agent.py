from memory import Memory
from model.Q_network import QNetwork

import torch
import torch.nn as nn


class MMAgent(nn.Module):
    def __init__(self, correct_reward, mask_reward, continuous_success_reward, connectivity_reward, detour_penalty):
        super(MMAgent, self).__init__()
        self.correct_reward = correct_reward
        self.mask_reward = mask_reward
        self.continuous_success_reward = continuous_success_reward
        self.connectivity_reward = connectivity_reward
        self.detour_penalty = detour_penalty
        self.main_net = QNetwork()
        self.target_net = QNetwork().eval()
        self.memory = Memory(100)
        # self.short_term_history_size = 3  # 短期历史的大小，可根据需要调整
        # self.short_term_history = None  # 初始化短期匹配历史的跟踪状态

    def select_action(self, last_traces_encoding, last_matched_road_segments_encoding, traces, matched_road_segments_id,
                      candidates, road_graph, trace_graph):
        with torch.no_grad():
            # 使用main_net进行前向传播，得到每个样本的动作Q值
            traces_encoding, matched_road_segments_encoding, action_values, _, _ = self.main_net(last_traces_encoding,
                                                                                           last_matched_road_segments_encoding,
                                                                                           traces,
                                                                                           matched_road_segments_id,
                                                                                           candidates, road_graph,
                                                                                           trace_graph)
            # 对每个样本选择最大Q值的动作
            return traces_encoding, matched_road_segments_encoding, torch.argmax(action_values, dim=-1)

    def reset_continuous_successes(self, batch_size):
        """重置连续成功次数的跟踪状态，每个批次开始时调用"""
        self.continuous_successes = torch.zeros(batch_size, dtype=torch.int32)

    def init_short_history(self, batch_size):
        """确保短期历史被初始化"""
        # self.short_term_history = [[] for _ in range(batch_size)]
        self.short_term_history = [[-1, -1] for _ in range(batch_size)]

    def update_short_term_history(self, matched_road_segments_id):
        """更新短期匹配历史，每次调用step函数时调用"""
        for i, road_id in enumerate(matched_road_segments_id):
            self.short_term_history[i].extend(road_id)
            # 保持历史长度不超过设置的短期历史大小
            self.short_term_history[i] = self.short_term_history[i][-self.short_term_history_size:]

    def step(self, last_matched_road, road_graph, action, candidates_id, tgt_roads, trace_lens, point_idx):
        seq_len = candidates_id.size(1)
        rewards = [[] for _ in range(seq_len)]  # 计算奖励的张量
        continuous_success_threshold = 3  # 连续成功的阈值，达到此阈值时给予额外奖励

        self.init_short_history(action.size(0))
        for i in range(action.size(0)):  # 首先遍历批次中的每个样本
            last_road_id = last_matched_road[i].item()
            for j in range(seq_len):  # 然后遍历样本的序列长度
                # 如果当前点的索引大于轨迹长度，则奖励为0
                if trace_lens[i] <= point_idx + 1 - (seq_len - 1) + j:
                    reward = self.mask_reward
                    self.continuous_successes[i] = 0  # 重置连续成功次数
                else:
                    # 从candidates_id中选择由action指定的候选路段ID
                    selected_candidate_id = candidates_id[i, j, action[i, j]]
                    # real_dis = road_graph.real_distance.get((last_road_id, selected_candidate_id.item()), -1)
                    # 更新短期历史逻辑
                    self.short_term_history[i][1] = self.short_term_history[i][0]  # 将上一次的结果移至第二位
                    self.short_term_history[i][0] = selected_candidate_id  # 将最新的结果放在第一位
                    # 根据选中的路段与目标路段是否相同给予奖励
                    # if selected_candidate_id == tgt_roads[i, j].item():
                    if action[i, j] == tgt_roads[i, j].item():
                        reward = self.correct_reward
                        self.continuous_successes[i] += 1  # 增加连续成功次数
                        # # 检查是否达到连续成功的阈值
                        if self.continuous_successes[i] >= continuous_success_threshold:
                            reward += self.continuous_success_reward  # 给予额外奖励
                    else:
                        reward = -self.correct_reward
                        # 连通性惩罚
                        connectivity = road_graph.connectivity_distances.get((last_road_id, selected_candidate_id.item()),-1)
                        if connectivity == -1:
                            reward += -self.connectivity_reward
                        elif connectivity > 2:
                            reward += -self.connectivity_reward/2
                        else:
                            reward += 0
                        self.continuous_successes[i] = 0  # 重置连续成功次数
                        # 绕路惩罚：检查当前匹配结果是否与前一结果不同且与再前一结果相同
                        if selected_candidate_id.item() != self.short_term_history[i][0] and \
                                selected_candidate_id.item() == self.short_term_history[i][1]:
                            reward += -self.detour_penalty  # 给予惩罚

                    # # 检查当前匹配道路是否在短期匹配历史中出现
                    # if selected_candidate_id.item() in self.short_term_history[i]:
                    #     reward -= short_term_penalty  # 给予惩罚

                    last_road_id = selected_candidate_id.item()

                rewards[j].append(reward)

        # 将奖励列表转换为张量
        return torch.tensor(rewards).transpose(0, 1)

    def update_target_net(self):
        # 更新目标网络参数，可以根据需要调整更新频率
        self.target_net.load_state_dict(self.main_net.state_dict())