from collections import namedtuple, deque
import random


# 定义一个新的 namedtuple，包括 trace, matched_road_segments_id, 和 candidates
State = namedtuple('State', (
    'traces_encoding',
    'matched_road_segments_encoding',
    'trace',
    'matched_road_segments_id',
    'candidates',
    'positive_samples',
    'negative_samples'
))

# 修改 Transition，以使用新的 State 结构
Transition = namedtuple('Transition', (
    'state',
    'action',
    'next_state',
    'reward'
))

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self,
             last_traces_encoding, last_matched_road_segments_encoding,
             trace, matched_road_segments_id, candidates,
             last_positive_samples, last_negative_samples,
             traces_encoding, matched_road_segments_encoding,
             next_trace, next_matched_road_segments_id, next_candidates,
             next_positive_samples, next_negative_samples,
             action, reward):
        # 创建当前状态的 namedtuple，包含正负样本
        state_namedtuple = State(last_traces_encoding, last_matched_road_segments_encoding,
                                 trace, matched_road_segments_id, candidates,
                                 last_positive_samples, last_negative_samples)

        # 创建下一个状态的 namedtuple，包含正负样本
        next_state_namedtuple = State(traces_encoding, matched_road_segments_encoding,
                                      next_trace, next_matched_road_segments_id, next_candidates,
                                      next_positive_samples, next_negative_samples)

        # 添加新的 Transition 到 memory
        self.memory.append(Transition(state_namedtuple, action, next_state_namedtuple, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
