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
from model.dqn import DQNAgent
from data_loader import MyDataset, padding
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
    training_episode = config['training_episode']
    learning_rate = config['learning_rate']

    train_batch_size = config['train_batch_size']
    test_batch_size = config['test_batch_size']
    train_size = config['train_size']
    test_size = config['test_size']
    optimize_batch_size = config['optimize_batch_size']

    gamma = config['gamma']
    target_update_interval = config['target_update_interval']
    downsample_rate = config['downsample_rate']
    match_interval = config['match_interval']

    correct_reward = config['correct_reward']
    mask_reward = config['mask_reward']

    return (training_episode, train_batch_size, test_batch_size, train_size, test_size, learning_rate, gamma,
            target_update_interval, downsample_rate, optimize_batch_size, match_interval, correct_reward, mask_reward)


def optimize_model(memory, dqn_agent, optimizer, road_graph, trace_graph, gamma, optimize_batch, match_interval, mask_reward):
    if len(memory) < optimize_batch:
        return

    transitions = memory.sample(optimize_batch)
    total_loss = 0.0

    for transition in transitions:
        # 解压单条记录中的 state, next_state, action, reward
        state, action, next_state, reward = transition
        reward = reward.to(device)
        mask = (reward != mask_reward).float()

        # 计算当前状态的 Q 值
        traces_encoding, matched_road_segments_encoding, trace, matched_road_segments_id, candidate = state
        _, _, q_values = dqn_agent.main_net(traces_encoding, matched_road_segments_encoding,
                                            trace, matched_road_segments_id, candidate, road_graph, trace_graph)

        state_action_values = q_values[:, 0, :].gather(-1, action[:, 0].unsqueeze(1))
        for i in range(1, match_interval):
            state_action_values = torch.cat(
                (state_action_values, q_values[:, i, :].gather(-1, action[:, i].unsqueeze(1))), dim=1)

        # Double DQN：分离动作选择和评估
        # 使用主网络选择下一个状态的最佳动作
        next_traces_encoding, next_matched_road_segments_encoding, next_trace, next_matched_road_segments_id, next_candidate = next_state
        _, _, q_values_next_main = dqn_agent.main_net(next_traces_encoding, next_matched_road_segments_encoding,
                                                      next_trace, next_matched_road_segments_id, next_candidate,
                                                      road_graph, trace_graph)
        max_next_action = q_values_next_main.max(-1)[1]  # (batch_size, match_interval)

        # 使用目标网络评估选择的动作的 Q 值
        _, _, q_values_next_target = dqn_agent.target_net(next_traces_encoding, next_matched_road_segments_encoding,
                                                          next_trace, next_matched_road_segments_id, next_candidate,
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

        # 计算Huber损失并累加
        loss = nn.SmoothL1Loss()(state_action_values * mask, expected_state_action_values * mask)
        total_loss += loss.item()

        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(transitions)
    return avg_loss


def train_agent(env, eval_env, agent, optimizer, road_graph, trace_graph, training_episode, gamma, target_update_interval,
                optimize_batch, match_interval, correct_reward, mask_reward, device):
    steps_done = 0
    best_acc = 0
    best_model = None
    early_stop_counter = 0
    early_stop_threshold = 20
    for episode in range(training_episode):
        env.reset()
        episode_reward = 0.0
        total_loss = 0.0
        update_steps = 0

        for batch in tqdm(range(env.num_of_batches)):
            data, done = env.step()
            if done:
                break

            traces, roads, candidates, candidates_id, trace_lens = data
            traces, roads, candidates_id = \
                traces.to(device), roads.to(device), candidates_id.to(device)
            matched_road_segments_id = torch.full((traces.size(0), match_interval, 1), -1).to(device)

            last_traces_encoding = None
            last_matched_road_segments_encoding = None
            # 从每条轨迹的第match_interval个点开始，每隔match_interval个点进行一次匹配
            for point_idx in range(match_interval - 1, traces.size(1) - match_interval, match_interval):
                # sub_traces = traces[:, point_idx + 1 - match_interval:point_idx + 1, :]
                sub_traces = traces[:, point_idx + 1 - match_interval:point_idx + 1]
                sub_candidates = candidates_id[:, point_idx + 1 - match_interval:point_idx + 1, :]

                traces_encoding, matched_road_segments_encoding, action = (
                    agent.select_action(last_traces_encoding, last_matched_road_segments_encoding,
                                        sub_traces, matched_road_segments_id, sub_candidates, road_graph, trace_graph))
                reward = agent.step(matched_road_segments_id[:,-1,:], road_graph, action, candidates_id[:, point_idx + 1 - match_interval:point_idx + 1, :],
                                    roads[:, point_idx + 1 - match_interval:point_idx + 1], trace_lens, point_idx)
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

                agent.memory.push(last_traces_encoding, last_matched_road_segments_encoding,
                                  sub_traces, matched_road_segments_id,
                                  sub_candidates,
                                  traces_encoding, matched_road_segments_encoding,
                                  traces[:, point_idx + 1:point_idx + 1 + match_interval],
                                  next_matched_road_segments_id,
                                  candidates_id[:, point_idx + 1:point_idx + 1 + match_interval, :],
                                  action, reward)

                matched_road_segments_id = next_matched_road_segments_id
                last_traces_encoding = traces_encoding
                last_matched_road_segments_encoding = matched_road_segments_encoding
                # last_traces_encoding = None
                # last_matched_road_segments_encoding = None

            # 更新损失和奖励
            loss = optimize_model(agent.memory, agent, optimizer, road_graph, trace_graph, gamma, optimize_batch, match_interval, mask_reward)
            steps_done += 1
            if steps_done % target_update_interval == 0:
                agent.update_target_net()

            if loss is not None:
                total_loss += loss
                update_steps += 1

        # 计算并打印每个episode的平均loss
        avg_loss = total_loss / update_steps if update_steps > 0 else 0
        print(f"Episode {episode}: Total Reward: {episode_reward}, Average Loss: {avg_loss}")
        acc = evaluate_agent(eval_env, agent, road_graph, trace_graph, match_interval, correct_reward, mask_reward, device)
        if best_acc < acc:
            best_model = deepcopy(agent)
            best_acc = acc
            early_stop_counter = 0  # 重置计数器，因为找到了更好的模型
        else:
            early_stop_counter += 1  # 没有改进，增加计数器
        if early_stop_counter >= early_stop_threshold:
            print(f"Early stopping triggered after {episode + 1} episodes.")
            break
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"best_model_{current_time}.pt"
    torch.save(best_model.state_dict(), filename)
    print(f"Best Accuracy: {best_acc}")


def evaluate_agent(env, agent, road_graph, trace_graph, match_interval, correct_reward, mask_reward, device):
    env.reset()
    accuracy_per_trace = 0
    cnt = 0
    for batch in tqdm(range(env.num_of_batches)):
        data, done = env.step()
        traces, roads, candidates, candidates_id, trace_lens = data
        traces, roads, candidates_id = \
            traces.to(device), roads.to(device), candidates_id.to(device)
        matched_road_segments_id = torch.full((traces.size(0), match_interval, 1), -1).to(device)

        last_traces_encoding = None
        last_matched_road_segments_encoding = None

        correct_counts = torch.zeros(traces.size(0))
        valid_counts = torch.zeros(traces.size(0))
        start_time = time.time()
        for point_idx in range(match_interval - 1, traces.size(1) - match_interval, match_interval):
            sub_traces = traces[:, point_idx + 1 - match_interval:point_idx + 1]
            sub_candidates = candidates_id[:, point_idx + 1 - match_interval:point_idx + 1, :]

            traces_encoding, matched_road_segments_encoding, action = (
                agent.select_action(last_traces_encoding, last_matched_road_segments_encoding,
                                    sub_traces, matched_road_segments_id, sub_candidates,
                                    road_graph, trace_graph))
            reward = agent.step(matched_road_segments_id[:, -1, :], road_graph, action, candidates_id[:, point_idx + 1 - match_interval:point_idx + 1, :],
                                roads[:, point_idx + 1 - match_interval:point_idx + 1], trace_lens, point_idx)

            # 更新匹配计数
            correct_counts += (reward == correct_reward).sum(1).float()
            valid_counts += (reward != mask_reward).sum(1).float()

            cur_matched_road_segments_id = (
                candidates_id[:, point_idx + 1 - match_interval, :].gather(-1, action[:, 0].unsqueeze(1)).unsqueeze(-1))
            for i in range(1, match_interval):
                cur_matched_road_segments_id = torch.cat((cur_matched_road_segments_id,
                                                          candidates_id[:, point_idx + 1 - match_interval + i, :]
                                                          .gather(-1, action[:, i].unsqueeze(1)).unsqueeze(-1)), dim=1)

            next_matched_road_segments_id = cur_matched_road_segments_id

            matched_road_segments_id = next_matched_road_segments_id
            last_traces_encoding = traces_encoding
            last_matched_road_segments_encoding = matched_road_segments_encoding
        end_time = time.time()
        print(f"执行时间：{end_time - start_time} 秒")

        # 计算每条轨迹的准确率并存储
        accuracy_per_trace += (correct_counts / valid_counts).mean()
        cnt += 1

    # print("matched traces:{}".format(cnt.sum()))
    average_accuracy = accuracy_per_trace.item()/cnt
    print(f"Average Accuracy: {average_accuracy}")
    return average_accuracy


if __name__ == '__main__':
    (training_episode, train_batch_size, test_batch_size, train_size, test_size, learning_rate, gamma,
     target_update_interval, downsample_rate, optimize_batch_size, match_interval, correct_reward,
     mask_reward) = loadConfig(config)

    data_path = osp.join('./data/data' + str(downsample_rate) + '_dis' + '/')
    road_graph = RoadGraph(root_path='./data', layer=4, gamma=10000, device=device)
    trace_graph = TraceGraph(data_path=data_path, device=device)

    train_set = MyDataset(path=data_path, name="train")
    test_set = MyDataset(path=data_path, name="test")

    train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, collate_fn=padding)
    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, collate_fn=padding)

    train_env = Environment(train_loader, math.ceil(train_size / train_batch_size))
    eval_env = Environment(test_loader, math.ceil(test_size / test_batch_size))

    print("loading dataset finished!")

    agent = DQNAgent(correct_reward, mask_reward)
    agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    print(agent)
    num_of_parameters = 0
    for name, parameters in agent.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)
    print("Starting training...")
    train_agent(train_env, eval_env, agent, optimizer, road_graph, trace_graph, training_episode, gamma, target_update_interval,
                optimize_batch_size, match_interval, correct_reward, mask_reward, device)