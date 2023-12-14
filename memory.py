from collections import namedtuple, deque
import random

# 定义一个新的 namedtuple，包括 trace, matched_road_segments_id, 和 candidates
State = namedtuple('State', ('trace', 'matched_road_segments_id', 'candidates'))

# 修改 Transition，以使用新的 State 结构
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, trace, matched_road_segments_id, candidates, action, next_trace, next_matched_road_segments_id, next_candidates, reward):
        # 使用 State namedtuple 创建 state 和 next_state
        state = State(trace, matched_road_segments_id, candidates)
        next_state = State(next_trace, next_matched_road_segments_id, next_candidates)
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
