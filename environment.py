
class Environment:
    def __init__(self, train_iter):
        # self.state_space_size = 2  # 状态空间维度
        # self.action_space_size = 2  # 动作空间维度
        self.train_iter = train_iter
        self.current_state = None

    def reset(self):
        # 重置环境，返回初始状态
        self.current_state = [0.0, 0.0]
        return self.current_state

    def step(self, action):
        # 执行动作，更新状态，返回下一个状态、奖励、是否终止等信息
        if action == 0:
            self.current_state[0] += 1.0
        else:
            self.current_state[1] -= 0.5

        # 模拟奖励（简化，实际根据具体问题定义）
        reward = self.current_state[0] - abs(self.current_state[1])

        # 判断是否终止
        done = abs(self.current_state[0]) > 5.0

        return self.current_state, reward, done