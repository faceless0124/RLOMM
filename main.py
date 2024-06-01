import argparse
import json
import os
import time
import math
from datetime import datetime

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import random
import os.path as osp
from copy import deepcopy

from tqdm import tqdm
from torch.utils.data import DataLoader
from model.mm_agent import MMAgent
from data_loader import MyDataset
from road_graph import RoadGraph
from trace_graph import TraceGraph
from environment import Environment

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--gpus", type=str, help="test program")

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def loadConfig(config):
    city = config['city']
    training_episode = config['training_episode']
    learning_rate = config['learning_rate']

    train_batch_size = config['train_batch_size']
    test_batch_size = config['test_batch_size']
    optimize_batch_size = config['optimize_batch_size']

    gamma = config['gamma']
    target_update_interval = config['target_update_interval']
    downsample_rate = config['downsample_rate']
    match_interval = config['match_interval']

    correct_reward = config['correct_reward']
    mask_reward = config['mask_reward']
    continuous_success_reward = config['continuous_success_reward']
    connectivity_reward = config['connectivity_reward']
    detour_penalty = config['detour_penalty']

    return (city, training_episode, train_batch_size, test_batch_size, learning_rate, gamma,
            target_update_interval, downsample_rate, optimize_batch_size, match_interval, correct_reward, mask_reward,
            continuous_success_reward, connectivity_reward, detour_penalty)


def contrastive_loss(features, positive_samples, negative_samples, temperature=0.5, eps=1e-8):
    batch_size, match_interval, feature_dim = features.shape

    # 重塑特征以计算相似度
    features = features.view(batch_size * match_interval, 1, feature_dim)
    positive_samples = positive_samples.view(batch_size * match_interval, feature_dim)
    negative_samples = negative_samples.view(batch_size * match_interval, -1, feature_dim)

    # 计算相似度
    positive_similarity = torch.sum(features * positive_samples.unsqueeze(1), dim=-1) / temperature
    negative_similarity = torch.bmm(negative_samples, features.transpose(1, 2)) / temperature

    # 应用数值稳定技巧：减去最大相似度
    max_similarity = torch.max(positive_similarity, torch.max(negative_similarity, dim=1).values)
    positive_similarity = positive_similarity - max_similarity
    negative_similarity = negative_similarity - max_similarity.unsqueeze(1)

    # 计算指数并防止溢出
    positive_similarity_exp = torch.exp(positive_similarity)
    negative_similarity_exp = torch.exp(negative_similarity).sum(dim=1)

    # 计算InfoNCE损失，确保分子远离零
    loss = -torch.log((positive_similarity_exp + eps) / (positive_similarity_exp + negative_similarity_exp + eps))

    return loss.mean()



def train_agent(train_env, valid_env, eval_env, agent, optimizer, road_graph, trace_graph, training_episode, gamma,
                target_update_interval,
                optimize_batch, match_interval, correct_reward, mask_reward, device):
    steps_done = 0
    best_acc = 0
    best_model = None
    early_stop_counter = 0
    early_stop_threshold = 20

    for episode in range(training_episode):
        train_env.reset()
        episode_reward = 0.0
        total_loss = 0.0
        total_rl_loss = 0.0
        total_ctr_loss = 0.0
        update_steps = 0
        all_time = 0.0

        for batch in tqdm(range(train_env.num_of_batches)):
            data, done = train_env.step()
            if done:
                break

            agent.reset_continuous_successes(data[0].size(0))  # 假设data[0]是traces，其第一维是批次大小

            traces, tgt_roads, candidates_id, trace_lens = data
            traces, tgt_roads, candidates_id = traces.to(device), tgt_roads.to(device), candidates_id.to(device)
            matched_road_segments_id = torch.full((traces.size(0), match_interval, 1), -1).to(device)

            last_traces_encoding = None
            last_matched_road_segments_encoding = None
            # 从每条轨迹的第match_interval个点开始，每隔match_interval个点进行一次匹配
            for point_idx in range(match_interval - 1, traces.size(1) - match_interval, match_interval):
                # sub_traces = traces[:, point_idx + 1 - match_interval:point_idx + 1, :]
                sub_traces = traces[:, point_idx + 1 - match_interval:point_idx + 1, :]
                sub_candidates = candidates_id[:, point_idx + 1 - match_interval:point_idx + 1, :]

                traces_encoding, matched_road_segments_encoding, action = (
                    agent.select_action(last_traces_encoding, last_matched_road_segments_encoding,
                                        sub_traces, matched_road_segments_id, sub_candidates, road_graph, trace_graph))

                reward = agent.step(matched_road_segments_id[:, -1, :], road_graph, action,
                                    candidates_id[:, point_idx + 1 - match_interval:point_idx + 1, :],
                                    tgt_roads[:, point_idx + 1 - match_interval:point_idx + 1], trace_lens, point_idx)
                episode_reward += reward.sum()

                cur_matched_road_segments_id = (
                    candidates_id[:, point_idx + 1 - match_interval, :].gather(-1, action[:, 0].unsqueeze(1)).unsqueeze(
                        -1))
                for i in range(1, match_interval):
                    cur_matched_road_segments_id = torch.cat((cur_matched_road_segments_id,
                                                              candidates_id[:, point_idx + 1 - match_interval + i, :]
                                                              .gather(-1, action[:, i].unsqueeze(1)).unsqueeze(-1)),
                                                             dim=1)

                next_matched_road_segments_id = cur_matched_road_segments_id

                current_positive_samples, current_negative_samples = get_positive_negative_samples(
                    tgt_roads[:, point_idx + 1 - match_interval:point_idx + 1],
                    candidates_id[:, point_idx + 1 - match_interval:point_idx + 1, :])
                # next_positive_samples, next_negative_samples = get_positive_negative_samples(
                #     roads[:, point_idx + 1:point_idx + 1 + match_interval], candidates_id[:, point_idx + 1:point_idx + 1 + match_interval, :])

                next_traces = traces[:, point_idx + 1:point_idx + 1 + match_interval, :]
                next_candidates = candidates_id[:, point_idx + 1:point_idx + 1 + match_interval, :]
                # agent.memory.push(
                #     last_traces_encoding, last_matched_road_segments_encoding,
                #     sub_traces, matched_road_segments_id, sub_candidates,
                #     None, None,  # 当前状态的正负样本
                #     traces_encoding, matched_road_segments_encoding,
                #     next_traces, next_matched_road_segments_id, next_candidates,
                #     None, None,  # 下一个状态的正负样本
                #     action, reward
                # )
                agent.memory.push(
                    last_traces_encoding, last_matched_road_segments_encoding,
                    sub_traces, matched_road_segments_id, sub_candidates,
                    current_positive_samples, current_negative_samples,  # 当前状态的正负样本
                    traces_encoding, matched_road_segments_encoding,
                    next_traces, next_matched_road_segments_id, next_candidates,
                    None, None,  # 下一个状态的正负样本
                    action, reward
                )

                matched_road_segments_id = next_matched_road_segments_id
                last_traces_encoding = traces_encoding
                last_matched_road_segments_encoding = matched_road_segments_encoding

            start_time = time.time()
            # 更新损失和奖励
            loss, rl_loss, ctr_loss = optimize_model(agent.memory, agent, optimizer, road_graph, trace_graph, gamma,
                                                     optimize_batch, match_interval, mask_reward, 0.1)
            end_time = time.time()
            all_time += end_time - start_time
            steps_done += 1
            if steps_done % target_update_interval == 0:
                agent.update_target_net()

            if loss is not None:
                total_loss += loss
                total_rl_loss += rl_loss
                total_ctr_loss += ctr_loss
                update_steps += 1
        print("training_time:", all_time)
        if update_steps == 0:
            continue
        # 计算并打印每个episode的平均loss
        avg_loss = total_loss / update_steps
        avg_rl_loss = total_rl_loss / update_steps
        avg_ctr_loss = total_ctr_loss / update_steps
        print(
            f"Episode {episode}: Total Reward: {episode_reward}, Average Loss: {avg_loss}, Average RL Loss: {avg_rl_loss}, Average CTR Loss: {avg_ctr_loss}")
        acc, rlcs = evaluate_agent(valid_env, agent, road_graph, trace_graph, match_interval, correct_reward,
                                   mask_reward, device, False)
        if best_acc < acc:
            best_model = deepcopy(agent.state_dict())
            best_acc = acc
            early_stop_counter = 0  # 重置计数器，因为找到了更好的模型
        else:
            early_stop_counter += 1  # 没有改进，增加计数器
            print(f"Early stopping counter: {early_stop_counter}")
        if early_stop_counter >= early_stop_threshold:
            print(f"Early stopping triggered after {episode + 1} episodes.")
            break
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./checkpoint/best_model_{current_time}.pt"
    torch.save(best_model, filename)
    test_agent = MMAgent(correct_reward, mask_reward, continuous_success_reward, connectivity_reward, detour_penalty)
    test_agent.load_state_dict(best_model)
    test_agent = test_agent.to(device)
    acc, rlcs = evaluate_agent(eval_env, test_agent, road_graph, trace_graph, match_interval, correct_reward,
                               mask_reward,
                               device, True)
    print(f"Best Accuracy: {acc} Best R-LCS: {rlcs}")


def optimize_model(memory, dqn_agent, optimizer, road_graph, trace_graph, gamma, optimize_batch, match_interval,
                   mask_reward, lambda_ctr):
    if len(memory) < optimize_batch:
        return None, None, None

    transitions = memory.sample(optimize_batch)
    total_loss = 0.0
    rl_loss_sum = 0.0
    ctr_loss_sum = 0.0

    for transition in transitions:
        # 解压单条记录中的 state, next_state, action, reward
        state, action, next_state, reward = transition
        reward = reward.to(device)
        mask = (reward != mask_reward).float()

        # 计算当前状态的 Q 值
        # traces_encoding, matched_road_segments_encoding, trace, matched_road_segments_id, candidate = state
        traces_encoding, matched_road_segments_encoding, trace, matched_road_segments_id, candidate, positive_samples, negative_samples = state
        _, _, q_values, road_emb, full_grid_emb = dqn_agent.main_net(traces_encoding, matched_road_segments_encoding,
                                                                     trace, matched_road_segments_id, candidate,
                                                                     road_graph, trace_graph)

        state_action_values = q_values[:, 0, :].gather(-1, action[:, 0].unsqueeze(1))
        for i in range(1, match_interval):
            state_action_values = torch.cat(
                (state_action_values, q_values[:, i, :].gather(-1, action[:, i].unsqueeze(1))), dim=1)

        # Double DQN：分离动作选择和评估
        # 使用主网络选择下一个状态的最佳动作
        next_traces_encoding, next_matched_road_segments_encoding, next_trace, next_matched_road_segments_id, next_candidate, _, _ = next_state
        _, _, q_values_next_main, _, _ = dqn_agent.main_net(next_traces_encoding, next_matched_road_segments_encoding,
                                                            next_trace, next_matched_road_segments_id, next_candidate,
                                                            road_graph, trace_graph)
        max_next_action = q_values_next_main.max(-1)[1]  # (batch_size, match_interval)

        # 使用目标网络评估选择的动作的 Q 值
        _, _, q_values_next_target, _, _ = dqn_agent.target_net(next_traces_encoding,
                                                                next_matched_road_segments_encoding,
                                                                next_trace, next_matched_road_segments_id,
                                                                next_candidate,
                                                                road_graph, trace_graph)

        next_state_values = q_values_next_target[:, 0, :].gather(-1, max_next_action[:, 0].unsqueeze(1)).detach()
        for i in range(1, match_interval):
            next_state_values = torch.cat((next_state_values, q_values_next_target[:, i, :].gather(-1,
                                                                                                   max_next_action[:,
                                                                                                   i].unsqueeze(
                                                                                                       1)).detach()),
                                          dim=1)
        # 计算期望 Q 值
        expected_state_action_values = (next_state_values * gamma) + reward

        # 计算强化学习损失
        rl_loss = nn.SmoothL1Loss()(state_action_values * mask, expected_state_action_values * mask)

        # postive_samples (batch_size, match_interval)
        # negative_samples (batch_size, match_interval, 9)
        positive_samples = positive_samples.to(device)
        negative_samples = negative_samples.to(device)
        features = full_grid_emb[trace[:, :, 0].to(torch.long)]
        positive_features = road_emb[positive_samples]
        negative_features = road_emb[negative_samples]

        # 计算对比学习损失
        ctr_loss = contrastive_loss(features * mask.unsqueeze(-1), positive_features * mask.unsqueeze(-1),
                                    negative_features * mask.unsqueeze(-1).unsqueeze(-1))

        # loss = rl_loss
        # total_loss += rl_loss.item()


        loss = rl_loss + lambda_ctr * ctr_loss
        total_loss += rl_loss.item() + lambda_ctr * ctr_loss.item()
        rl_loss_sum += rl_loss.item()
        ctr_loss_sum += lambda_ctr * ctr_loss.item()

        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(transitions)
    avg_rl_loss = rl_loss_sum / len(transitions)
    avg_ctr_loss = ctr_loss_sum / len(transitions)
    return avg_loss, avg_rl_loss, avg_ctr_loss

def get_positive_negative_samples(roads_slice, candidates_slice):
    batch_size, seq_len, _ = candidates_slice.shape  # 获取维度信息
    # 获取正样本：直接根据roads_slice中的索引从candidates_slice中提取
    positive_samples = torch.gather(candidates_slice, 2, roads_slice.unsqueeze(-1)).squeeze(-1)
    # 初始化一个tensor，用于存放负样本的索引
    all_indices = torch.arange(candidates_slice.shape[2], device=candidates_slice.device).expand(batch_size, seq_len, -1)
    # 创建一个mask，其中正样本的位置为False，其余为True
    mask = all_indices != roads_slice.unsqueeze(-1)
    # 应用mask，选择负样本
    negative_samples = torch.masked_select(candidates_slice, mask).view(batch_size, seq_len, -1)
    return positive_samples, negative_samples


def longest_common_subsequence(seq1, seq2):
    # 创建一个二维数组，用于存储两个序列的LCS长度
    lcs_matrix = [[0 for _ in range(len(seq2) + 1)] for _ in range(len(seq1) + 1)]

    # 动态规划填充这个矩阵
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])

    # 矩阵的最后一个元素包含了LCS的长度
    return lcs_matrix[-1][-1]


def evaluate_agent(env, agent, road_graph, trace_graph, match_interval, correct_reward, mask_reward, device, is_rlcs):
    env.reset()
    eval_acc_sum = 0
    eval_rlcs_sum = 0
    cnt = 0
    all_time = 0.
    for batch in tqdm(range(env.num_of_batches)):
        data, done = env.step()

        agent.reset_continuous_successes(data[0].size(0))  # 假设data[0]是traces，其第一维是批次大小

        traces, roads, candidates_id, trace_lens = data
        traces, roads, candidates_id = \
            traces.to(device), roads.to(device), candidates_id.to(device)
        matched_road_segments_id = torch.full((traces.size(0), match_interval, 1), -1).to(device)

        last_traces_encoding = None
        last_matched_road_segments_encoding = None

        trace_matched_id = []
        trace_ground_truth = []
        # 初始化用于收集匹配结果和ground truth的列表
        for point_idx in range(match_interval - 1, traces.size(1) - match_interval, match_interval):
            sub_traces = traces[:, point_idx + 1 - match_interval:point_idx + 1, :]
            sub_candidates = candidates_id[:, point_idx + 1 - match_interval:point_idx + 1, :]
            start_time = time.time()
            traces_encoding, matched_road_segments_encoding, action = (
                agent.select_action(last_traces_encoding, last_matched_road_segments_encoding,
                                    sub_traces, matched_road_segments_id, sub_candidates,
                                    road_graph, trace_graph))
            end_time = time.time()
            all_time += end_time - start_time

            cur_matched_road_segments_id = candidates_id[:, point_idx + 1 - match_interval, :].gather(-1, action[:,
                                                                                                          0].unsqueeze(
                1)).unsqueeze(-1)
            for matched_id in range(1, match_interval):
                cur_matched_road_segments_id = torch.cat((cur_matched_road_segments_id,
                                                          candidates_id[:, point_idx + 1 - match_interval + matched_id,
                                                          :]
                                                          .gather(-1, action[:, matched_id].unsqueeze(1)).unsqueeze(
                                                              -1)), dim=1)
            trace_matched_id.append(cur_matched_road_segments_id)

            cur_trace_ground_truth = candidates_id[:, point_idx + 1 - match_interval, :].gather(-1, roads[:,
                                                                                                    point_idx + 1 - match_interval].unsqueeze(
                1)).unsqueeze(-1)
            for matched_id in range(1, match_interval):
                cur_trace_ground_truth = torch.cat((cur_trace_ground_truth,
                                                    candidates_id[:, point_idx + 1 - match_interval + matched_id, :]
                                                    .gather(-1, roads[:,
                                                                point_idx + 1 - match_interval + matched_id].unsqueeze(
                                                        1)).unsqueeze(-1)), dim=1)
            # trace_ground_truth.append(roads[:, point_idx + 1 - match_interval:point_idx + 1])
            trace_ground_truth.append(cur_trace_ground_truth)

            matched_road_segments_id = cur_matched_road_segments_id
            last_traces_encoding = traces_encoding
            last_matched_road_segments_encoding = matched_road_segments_encoding

        # 对列表中的元素沿着match_interval维度进行拼接
        trace_matched_id = torch.cat(trace_matched_id, dim=1).squeeze(-1)  # 拼接所有批次的匹配ID
        trace_ground_truth = torch.cat(trace_ground_truth, dim=1).squeeze(-1)  # 拼接所有批次的真实路段ID

        masks = torch.zeros_like(trace_matched_id, dtype=torch.bool)
        max_len = masks.size(1)
        # 遍历每个批次的轨迹长度，并将小于等于轨迹长度的部分设置为True
        for batch_idx, trace_len in enumerate(trace_lens):
            idx = min(trace_len, max_len)
            masks[batch_idx, :idx] = True

        r_lcs_values = []
        acc_values = []
        for matched_id, ground_truth, mask in zip(trace_matched_id, trace_ground_truth, masks):
            matched_id = matched_id[mask]
            ground_truth = ground_truth[mask]
            if is_rlcs:
                lcs_length = longest_common_subsequence(matched_id, ground_truth)
            else:
                lcs_length = 0
            r_lcs_values.append(lcs_length / len(matched_id))

            acc_values.append((matched_id == ground_truth).sum() / len(matched_id))
        average_r_lcs = sum(r_lcs_values) / len(r_lcs_values)
        eval_rlcs_sum += average_r_lcs
        average_acc = sum(acc_values) / len(acc_values)
        eval_acc_sum += average_acc
        cnt += 1
    average_accuracy = eval_acc_sum / cnt
    average_r_lcs = eval_rlcs_sum / cnt
    print(f"Average Accuracy: {average_accuracy} Average R-LCS: {average_r_lcs}")
    print(f"all time: {all_time}")
    return average_accuracy, average_r_lcs


if __name__ == '__main__':
    (city, training_episode, train_batch_size, test_batch_size, learning_rate, gamma,
     target_update_interval, downsample_rate, optimize_batch_size, match_interval, correct_reward,
     mask_reward, continuous_success_reward, connectivity_reward, detour_penalty) = loadConfig(config)

    data_path = osp.join('./data/' + city + '/data' + str(downsample_rate) + '_dis' + '/')
    road_graph = RoadGraph(root_path='./data/' + city, layer=4, gamma=10000, device=device)
    trace_graph = TraceGraph(data_path=data_path, device=device)

    train_set = MyDataset(path=data_path, name="train", city=city)
    valid_set = MyDataset(path=data_path, name="val", city=city)
    test_set = MyDataset(path=data_path, name="test", city=city)

    train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, collate_fn=train_set.padding)
    valid_loader = DataLoader(dataset=valid_set, batch_size=test_batch_size, collate_fn=valid_set.padding)
    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, collate_fn=test_set.padding)

    train_env = Environment(train_loader, math.ceil(len(train_set) / train_batch_size))
    valid_env = Environment(valid_loader, math.ceil(len(valid_set) / test_batch_size))
    eval_env = Environment(test_loader, math.ceil(len(test_set) / test_batch_size))

    print("loading dataset finished!")

    agent = MMAgent(correct_reward, mask_reward, continuous_success_reward, connectivity_reward, detour_penalty)
    agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    # print(agent)
    num_of_parameters = 0
    for name, parameters in agent.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)
    print("Starting training...")
    train_agent(train_env, valid_env, eval_env, agent, optimizer, road_graph, trace_graph, training_episode, gamma,
                target_update_interval, optimize_batch_size, match_interval, correct_reward, mask_reward, device)
