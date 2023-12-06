import argparse
import json
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import random
import os.path as osp
from memory import Transition
from torch.utils.data import DataLoader
from dqn import DQNAgent
from environment import Environment
from data_loader import MyDataset, padding

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
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    gamma = config['gamma']
    target_update_interval = config['target_update_interval']

    downsample_rate = config['downsample_rate']

    return training_episode, batch_size, learning_rate, gamma, target_update_interval, downsample_rate


def optimize_model(memory, dqn_agent, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接batch元素
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t, a) - 模型计算的Q值
    state_action_values = dqn_agent.policy_net(state_batch).gather(1, action_batch)

    # 计算下一个状态的V(s_{t+1})
    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = dqn_agent.target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # 计算Huber损失
    loss = nn.SmoothL1Loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    training_episode, batch_size, learning_rate, gamma, target_update_interval, downsample_rate = loadConfig(config)

    data_path = osp.join('./data/data' + str(downsample_rate) + '/')
    train_set = MyDataset( path=data_path, name="train")
    val_set = MyDataset(path=data_path, name="val")
    test_set = MyDataset(path=data_path, name="test")

    train_iter = DataLoader(dataset=train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=padding)
    val_iter = DataLoader(dataset=val_set,
                          batch_size=batch_size,
                          collate_fn=padding)
    test_iter = DataLoader(dataset=test_set,
                           batch_size=batch_size,
                           collate_fn=padding)

    print("loading dataset finished!")
    for data in train_iter:
        traces = data[0]
        tgt_roads = data[1]
        candidates = data[2]
        sample_Idx = data[3]
        print(traces)
        print("***********************************")
        print(tgt_roads)
        print("***********************************")
        print(candidates)
        print("***********************************")
        print(sample_Idx)
        exit(0)
    state_dim = 10  # 状态维度
    action_dim = 4  # 动作维度
    agent = DQNAgent(state_dim, action_dim)

    agent.to(device)
    print(agent)

    num_of_parameters = 0
    for name, parameters in agent.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # TODO：修改训练过程
    steps_done = 0
    env = Environment()
    # 与环境交互，收集经验
    for episode in range(training_episode):
        state = env.reset()  # 重置环境，获取初始状态
        done = False  # 确保 done 被重置

        while not done:  # 假设每个episode有10个时间步
            action = agent.select_action(state, steps_done)
            steps_done += 1
            print("Selected Action:", action.item())
            next_state, reward, done, _ = env.step(action.item())
            # 存储经验
            agent.memory.append(Transition(state, action, next_state, reward))
            # 更新状态
            state = next_state
            # 优化模型
            optimize_model(agent.memory, agent, optimizer, batch_size=batch_size, gamma=gamma)
            # 定期更新目标网络
            if steps_done % target_update_interval == 0:
                agent.update_target_net()
