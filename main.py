import argparse
import json
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import random
import os.path as osp

from tqdm import tqdm

from memory import Transition
from torch.utils.data import DataLoader
from model.dqn import DQNAgent
from data_loader import MyDataset, padding
from road_graph import RoadGraph
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
    test_episode = config['test_episode']

    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    gamma = config['gamma']
    target_update_interval = config['target_update_interval']

    downsample_rate = config['downsample_rate']

    return training_episode, test_episode, batch_size, learning_rate, gamma, target_update_interval, downsample_rate


# def optimize_model(memory, dqn_agent, optimizer, road_graph, gamma):
#     if len(memory) < 1:
#         return
#
#     transitions = memory.sample(1)
#     batch = Transition(*zip(*transitions))
#
#     # 解压 state, next_state, action_batch, reward_batch
#     states = batch.state[0]
#     next_states = batch.next_state[0]
#     action_batch = batch.action[0]
#     reward_batch = batch.reward[0].unsqueeze(1).to(device)
#
#     mask = (reward_batch != 0).float()
#
#     # TODO 修改为Double DQN
#     trace, matched_road_segments_id, candidate = states
#     q_values = dqn_agent.main_net(trace, matched_road_segments_id, candidate, road_graph, trace.size(1))
#     state_action_values = q_values.gather(1, action_batch.unsqueeze(1)) # 选择对应动作的 Q 值
#
#     next_trace, next_matched_road_segments_id, next_candidate = next_states
#     q_values_next = dqn_agent.target_net(next_trace, next_matched_road_segments_id, next_candidate, road_graph, trace.size(1))
#     next_state_values = q_values_next.max(1)[0].detach().unsqueeze(1) # 选择最大 Q 值
#
#     expected_state_action_values = (next_state_values * gamma) + reward_batch
#
#     # 计算Huber损失
#     loss = nn.SmoothL1Loss()(state_action_values * mask, expected_state_action_values * mask)
#     # 优化模型
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss.item()  # 返回损失值

def optimize_model(memory, dqn_agent, optimizer, road_graph, gamma):
    if len(memory) < 32:
        return

    transitions = memory.sample(32)
    total_loss = 0.0

    for transition in transitions:
        # 解压单条记录中的 state, next_state, action, reward
        state, action, next_state, reward = transition
        reward = reward.unsqueeze(1).to(device)
        action = action.unsqueeze(1)
        mask = (reward != 0).float()

        # 计算当前状态的 Q 值
        trace, matched_road_segments_id, candidate = state
        q_values = dqn_agent.main_net(trace, matched_road_segments_id, candidate, road_graph, trace.size(1))
        state_action_values = q_values.gather(1, action)

        # 计算下一个状态的最大预期 Q 值
        next_trace, next_matched_road_segments_id, next_candidate = next_state
        q_values_next = dqn_agent.target_net(next_trace, next_matched_road_segments_id, next_candidate, road_graph, trace.size(1))
        next_state_values = q_values_next.max(1)[0].detach().unsqueeze(1)

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




def train_agent(env, eval_env, agent, optimizer, road_graph, training_episode, gamma, target_update_interval, device):
    steps_done = 0
    for episode in range(training_episode):
        env.reset()
        episode_reward = 0
        total_loss = 0.0
        update_steps = 0

        for batch in tqdm(range(env.num_of_batches)):
            data, done = env.step()
            if done:
                break

            traces, roads, candidates, candidates_id, trace_lens = data

            traces, roads, candidates, candidates_id = \
                traces.to(device), roads.to(device), candidates.to(device), candidates_id.to(device)
            matched_road_segments_id = torch.full((traces.size(0), 1, 1), -1).to(device)

            for point_idx in range(traces.size(1)-1):
                sub_traces = traces[:, :point_idx + 1, :]
                # sub_candidates = candidates[:, point_idx, :, :]
                sub_candidates = candidates_id[:, point_idx, :]

                action = agent.select_action(sub_traces, matched_road_segments_id, sub_candidates, road_graph, point_idx)
                reward = agent.step(action, candidates_id[:, point_idx, :], roads[:, point_idx], trace_lens, point_idx)

                episode_reward += reward.sum()

                cur_matched_road_segments_id = candidates_id[:, point_idx, :].gather(1, action.unsqueeze(1)).unsqueeze(-1)
                next_matched_road_segments_id = torch.cat((matched_road_segments_id, cur_matched_road_segments_id),
                                                          dim=1)


                agent.memory.push(traces[:, :point_idx + 1, :], matched_road_segments_id, candidates_id[:, point_idx, :],
                                      traces[:, :point_idx + 2, :], next_matched_road_segments_id, candidates_id[:, point_idx + 1, :],
                                      action, reward)

                matched_road_segments_id = next_matched_road_segments_id

                steps_done += 1
                if steps_done % target_update_interval == 0:
                    agent.update_target_net()

            # 更新损失和奖励
            loss = optimize_model(agent.memory, agent, optimizer, road_graph, gamma)
            if loss is not None:
                total_loss += loss
                update_steps += 1

        # 计算并打印每个episode的平均loss
        avg_loss = total_loss / update_steps if update_steps > 0 else 0
        print(f"Episode {episode}: Total Reward: {episode_reward}, Average Loss: {avg_loss}")
        evaluate_agent(eval_env, agent, road_graph, device)



def evaluate_agent(env, agent, road_graph, device):
    env.reset()
    data, done = env.step()
    traces, roads, candidates, candidates_id, trace_lens = data
    traces, roads, candidates, candidates_id = \
        traces.to(device), roads.to(device), candidates.to(device), candidates_id.to(device)
    matched_road_segments_id = torch.full((traces.size(0), 1, 1), -1).to(device)

    correct_counts = torch.zeros(traces.size(0))
    valid_counts = torch.zeros(traces.size(0))

    for point_idx in tqdm(range(traces.size(1))):
        sub_traces = traces[:, :point_idx + 1, :]
        # sub_candidates = candidates[:, point_idx, :, :]
        sub_candidates = candidates_id[:, point_idx, :]

        action = agent.select_action(sub_traces, matched_road_segments_id, sub_candidates, road_graph, point_idx)
        reward = agent.step(action, candidates_id[:, point_idx, :], roads[:, point_idx], trace_lens, point_idx)

        # 更新匹配计数
        correct_counts += (reward == 10).float()
        valid_counts += (reward != 0).float()

        cur_matched_road_segments_id = candidates_id[:, point_idx, :].gather(1, action.unsqueeze(1)).unsqueeze(-1)
        next_matched_road_segments_id = torch.cat((matched_road_segments_id, cur_matched_road_segments_id), dim=1)

        matched_road_segments_id = next_matched_road_segments_id

    # 计算每条轨迹的准确率并存储
    accuracy_per_trace = correct_counts / valid_counts
    cnt = (correct_counts != 0)

    print(torch.sum(cnt))
    average_accuracy = accuracy_per_trace.mean()

    print(f"Average Accuracy: {average_accuracy.item()}")
    return average_accuracy.item()



if __name__ == '__main__':
    training_episode, test_episode, batch_size, learning_rate, gamma, target_update_interval, downsample_rate = loadConfig(config)

    data_path = osp.join('./data/data' + str(downsample_rate) + '_+timestamp' + '/')
    road_graph = RoadGraph(root_path='./data',
                           layer=4,
                           gamma=10000,
                           device=device)

    train_set = MyDataset(path=data_path, name="train")
    test_set = MyDataset(path=data_path, name="test")

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=padding)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=3000,
                             collate_fn=padding)

    train_env = Environment(train_loader, batch_size)
    eval_env = Environment(test_loader, batch_size)

    print("loading dataset finished!")

    agent = DQNAgent()
    agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    print(agent)
    num_of_parameters = 0
    for name, parameters in agent.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)
    print("Starting training...")
    train_agent(train_env, eval_env, agent, optimizer, road_graph, training_episode, gamma, target_update_interval, device)