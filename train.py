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


def optimize_model(memory, dqn_agent, optimizer, road_graph, gamma):
    if len(memory) < 1:
        return

    transitions = memory.sample(1)
    batch = Transition(*zip(*transitions))

    # 解压 state, next_state, action_batch, reward_batch
    states = batch.state[0]
    next_states = batch.next_state[0]
    action_batch = batch.action[0]
    reward_batch = batch.reward[0].unsqueeze(1).to(device)

    trace, matched_road_segments_id, candidate = states
    q_values = dqn_agent.policy_net(trace, matched_road_segments_id, candidate, road_graph, trace.size(1))
    state_action_values = q_values.gather(1, action_batch.unsqueeze(1)) # 选择对应动作的 Q 值

    next_trace, next_matched_road_segments_id, next_candidate = next_states
    q_values_next = dqn_agent.target_net(next_trace, next_matched_road_segments_id, next_candidate, road_graph, trace.size(1))
    next_state_values = q_values_next.max(1)[0].detach().unsqueeze(1) # 选择最大 Q 值

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # 计算Huber损失
    loss = nn.SmoothL1Loss()(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))
    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()  # 返回损失值


def train_agent(env, agent, optimizer, road_graph, training_episode, gamma, target_update_interval, device):
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

            traces, tgt_roads, candidates, candidates_id, trace_lens = data

            traces = torch.FloatTensor(traces).to(device)
            tgt_roads = torch.LongTensor(tgt_roads).to(device)
            candidates = torch.FloatTensor(candidates).to(device)
            candidates_id = torch.LongTensor(candidates_id).to(device)
            matched_road_segments_id = torch.full((traces.size(0), 1, 1), -1).to(device)

            for i in range(traces.size(1)):
                sub_traces = traces[:, :i + 1, :]
                sub_candidates = candidates[:, i, :, :]

                action = agent.select_action(sub_traces, matched_road_segments_id, sub_candidates, road_graph, trace_lens, i)
                reward = agent.step(action, candidates_id[:, i, :], tgt_roads[:, i])

                episode_reward += reward.sum()

                cur_matched_road_segments_id = candidates_id[:, i, :].gather(1, action.unsqueeze(1)).unsqueeze(-1)
                next_matched_road_segments_id = torch.cat((matched_road_segments_id, cur_matched_road_segments_id),
                                                          dim=1)

                if i == traces.size(1) - 1:
                    continue
                    # agent.memory.push(traces[:, :i + 1, :], matched_road_segments_id[:, :, :], candidates[:, i, :, :],
                    #                   None, None,
                    #                   None, action, reward)
                else:
                    agent.memory.push(traces[:, :i + 1, :], matched_road_segments_id, candidates[:, i, :, :],
                                      traces[:, :i + 2, :], next_matched_road_segments_id, candidates[:, i + 1, :, :],
                                      action, reward)

                matched_road_segments_id = next_matched_road_segments_id

                # 更新损失和奖励
                loss = optimize_model(agent.memory, agent, optimizer, road_graph, gamma)
                if loss is not None:
                    total_loss += loss
                    update_steps += 1

                steps_done += 1
                if steps_done % target_update_interval == 0:
                    agent.update_target_net()

        # 计算并打印每个episode的平均loss
        avg_loss = total_loss / update_steps if update_steps > 0 else 0
        print(f"Episode {episode}: Total Reward: {episode_reward}, Average Loss: {avg_loss}")


def evaluate_agent(env, agent, road_graph, episodes, device):
    total_rewards = 0.0

    for episode in range(episodes):
        env.reset()
        episode_reward = 0

        while True:
            data, done = env.step()
            if done:
                break

            traces, tgt_roads, candidates, candidates_id, trace_lens = data

            traces = torch.FloatTensor(traces).to(device)
            tgt_roads = torch.LongTensor(tgt_roads).to(device)
            candidates = torch.FloatTensor(candidates).to(device)
            candidates_id = torch.LongTensor(candidates_id).to(device)
            matched_road_segments_id = torch.full((traces.size(0), 1, 1), -1).to(device)

            for i in range(traces.size(1)):
                sub_traces = traces[:, :i + 1, :]
                sub_candidates = candidates[:, i, :, :]

                action = agent.select_action(sub_traces, matched_road_segments_id, sub_candidates, road_graph, trace_lens, i)
                reward = agent.step(action, candidates_id[:, i, :], tgt_roads[:, i])
                episode_reward += reward.sum()

                cur_matched_road_segments_id = candidates_id[:, i, :].gather(1, action.unsqueeze(1)).unsqueeze(-1)
                next_matched_road_segments_id = torch.cat((matched_road_segments_id, cur_matched_road_segments_id), dim=1)

                matched_road_segments_id = next_matched_road_segments_id

        total_rewards += episode_reward
        print(f"Episode {episode}: Reward: {episode_reward}")

    avg_reward = total_rewards / episodes
    print(f"Average Reward: {avg_reward}")
    return avg_reward



if __name__ == '__main__':
    training_episode, test_episode, batch_size, learning_rate, gamma, target_update_interval, downsample_rate = loadConfig(config)

    data_path = osp.join('./data/data' + str(downsample_rate) + '/')
    road_graph = RoadGraph(root_path='./data',
                           layer=4,
                           gamma=10000,
                           device=device)

    train_set = MyDataset(path=data_path, name="train")
    test_set = MyDataset(path=data_path, name="test")

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=padding)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             collate_fn=padding)

    train_env = Environment(train_loader)
    eval_env = Environment(test_loader)

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
    train_agent(train_env, agent, optimizer, road_graph, training_episode, gamma, target_update_interval,
                device)

    print("Starting evaluation...")
    evaluate_agent(eval_env, agent, road_graph, test_episode, device)