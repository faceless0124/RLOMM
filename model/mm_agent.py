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
        # self.short_term_history_size = 3
        # self.short_term_history = None

    def select_action(self, last_traces_encoding, last_matched_road_segments_encoding, traces, matched_road_segments_id,
                      candidates, road_graph, trace_graph):
        with torch.no_grad():
            traces_encoding, matched_road_segments_encoding, action_values, _, _ = self.main_net(last_traces_encoding,
                                                                                           last_matched_road_segments_encoding,
                                                                                           traces,
                                                                                           matched_road_segments_id,
                                                                                           candidates, road_graph,
                                                                                           trace_graph)
            return traces_encoding, matched_road_segments_encoding, torch.argmax(action_values, dim=-1)

    def reset_continuous_successes(self, batch_size):
        self.continuous_successes = torch.zeros(batch_size, dtype=torch.int32)

    def init_short_history(self, batch_size):
        # self.short_term_history = [[] for _ in range(batch_size)]
        self.short_term_history = [[-1, -1] for _ in range(batch_size)]

    def update_short_term_history(self, matched_road_segments_id):
        for i, road_id in enumerate(matched_road_segments_id):
            self.short_term_history[i].extend(road_id)
            self.short_term_history[i] = self.short_term_history[i][-self.short_term_history_size:]

    def step(self, last_matched_road, road_graph, action, candidates_id, tgt_roads, trace_lens, point_idx):
        seq_len = candidates_id.size(1)
        rewards = [[] for _ in range(seq_len)]
        continuous_success_threshold = 3

        self.init_short_history(action.size(0))
        for i in range(action.size(0)):
            last_road_id = last_matched_road[i].item()
            for j in range(seq_len):
                if trace_lens[i] <= point_idx + 1 - (seq_len - 1) + j:
                    reward = self.mask_reward
                    self.continuous_successes[i] = 0
                else:
                    selected_candidate_id = candidates_id[i, j, action[i, j]]
                    # real_dis = road_graph.real_distance.get((last_road_id, selected_candidate_id.item()), -1)
                    self.short_term_history[i][1] = self.short_term_history[i][0]
                    self.short_term_history[i][0] = selected_candidate_id
                    # if selected_candidate_id == tgt_roads[i, j].item():
                    if action[i, j] == tgt_roads[i, j].item():
                        reward = self.correct_reward
                        self.continuous_successes[i] += 1
                        if self.continuous_successes[i] >= continuous_success_threshold:
                            reward += self.continuous_success_reward
                    else:
                        reward = -self.correct_reward
                        connectivity = road_graph.connectivity_distances.get((last_road_id, selected_candidate_id.item()),-1)
                        if connectivity == -1:
                            reward += -self.connectivity_reward
                        elif connectivity > 2:
                            reward += -self.connectivity_reward/2
                        else:
                            reward += 0
                        self.continuous_successes[i] = 0
                        if selected_candidate_id.item() != self.short_term_history[i][0] and \
                                selected_candidate_id.item() == self.short_term_history[i][1]:
                            reward += -self.detour_penalty


                    last_road_id = selected_candidate_id.item()

                rewards[j].append(reward)

        return torch.tensor(rewards).transpose(0, 1)

    def update_target_net(self):
        self.target_net.load_state_dict(self.main_net.state_dict())