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
from model.dqn import DQNAgent
from data_loader import MyDataset, padding
from road_graph import RoadGraph

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

    # 解压 state 和 next_state
    states = [s.state for s in transitions]
    next_states = [s.next_state for s in transitions]

    # 创建一个掩码，标记非最终状态
    non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)

    # 只处理非最终状态的 next_states
    non_final_next_states = [s for s in next_states if s is not None]

    # 对 states 和 non_final_next_states 中的状态进行处理
    traces = torch.cat([s.trace for s in states])
    matched_road_segments_ids = torch.cat([s.matched_road_segments_id for s in states])
    candidates = torch.cat([s.candidates for s in states])

    if non_final_next_states:
        non_final_traces = torch.cat([s.trace for s in non_final_next_states])
        non_final_matched_road_segments_ids = torch.cat([s.matched_road_segments_id for s in non_final_next_states])
        non_final_candidates = torch.cat([s.candidates for s in non_final_next_states])

    # 把 action 和 reward 转换成张量
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t, a) - 模型计算的Q值
    # 注意：需要修改 policy_net 来接受新的输入格式
    state_action_values = dqn_agent.policy_net(traces, matched_road_segments_ids, candidates).gather(1, action_batch)

    # 计算下一个状态的V(s_{t+1})
    next_state_values = torch.zeros(batch_size)
    if non_final_next_states:
        # 注意：需要修改 target_net 来接受新的输入格式
        next_state_values[non_final_mask] = \
        dqn_agent.target_net(non_final_traces, non_final_matched_road_segments_ids, non_final_candidates).max(1)[
            0].detach()

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # 计算Huber损失
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # 返回损失值

if __name__ == '__main__':
    training_episode, batch_size, learning_rate, gamma, target_update_interval, downsample_rate = loadConfig(config)

    data_path = osp.join('./data/data' + str(downsample_rate) + '/')
    road_graph = RoadGraph(root_path='./data',
                           layer=4,
                           gamma=10000,
                           device=device)

    train_set = MyDataset(path=data_path, name="train")
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

    state_dim = 10  # 状态维度
    action_dim = 14  # 动作维度
    agent = DQNAgent(action_dim)

    agent.to(device)
    print(agent)

    num_of_parameters = 0
    for name, parameters in agent.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # Training loop
    for data in train_iter:
        traces = data[0].to(device)
        tgt_roads = data[1].to(device)
        candidates = data[2].to(device)
        candidates_id = data[3].to(device)

        matched_road_segments_id = torch.tensor([-1]).to(device)
        # matched_road_segments_id = torch.tensor([2,5,7,98])
        for i in range(traces.size(0)):
            steps_done = 0
            trace = traces[i][0].unsqueeze(0)
            candidate = candidates[i][0]
            for j in range(traces.size(1)):
                steps_done += 1

                action = agent.select_action(trace, matched_road_segments_id, road_graph, candidate, steps_done)
                print("select action {}".format(action))
                # Take the action and get the next state and reward
                reward = agent.step(action, candidates_id[i][j], tgt_roads[i][j])
                print("reward:", reward)
                if j == traces.size(1) - 1:
                    next_trace = None
                    next_matched_road_segments_id = None
                    next_candidate = None
                else:
                    next_trace = torch.cat((trace, traces[i][j + 1].unsqueeze(0)), dim=0)
                    next_matched_road_segments_id = torch.cat(
                        (matched_road_segments_id, candidates_id[i][j][action].view(1)), dim=0)
                    next_candidate = candidates[i][j + 1]
                # Store the transition in the replay memory
                agent.memory.push(trace, matched_road_segments_id, candidates_id[i][j], action, next_trace,
                                  next_matched_road_segments_id, next_candidate, reward)

                trace = next_trace
                matched_road_segments_id = next_matched_road_segments_id
                candidate = next_candidate

        # Optimize the model using samples from the replay memory in batches
        loss = optimize_model(agent.memory, agent, optimizer, batch_size, gamma)
        print("loss:", loss)

    ### 使用environment的版本
    # steps_done = 0
    # env = Environment(train_iter)
    # # 与环境交互，收集经验
    # for episode in range(training_episode):
    #     state = env.reset()  # 重置环境，获取初始状态
    #     done = False  # 确保 done 被重置
    #
    #     while not done:  # 假设每个episode有10个时间步
    #         action = agent.select_action(state, steps_done)
    #         steps_done += 1
    #         print("Selected Action:", action.item())
    #         next_state, reward, done, _ = env.step(action.item())
    #         # 存储经验
    #         agent.memory.append(Transition(state, action, next_state, reward))
    #         # 更新状态
    #         state = next_state
    #         # 优化模型
    #         optimize_model(agent.memory, agent, optimizer, batch_size=batch_size, gamma=gamma)
    #         # 定期更新目标网络
    #         if steps_done % target_update_interval == 0:
    #             agent.update_target_net()
