from collections import namedtuple, deque
import random

# # 定义一个新的 namedtuple，包括 trace, matched_road_segments_id, 和 candidates
# State = namedtuple('State', ('trace', 'matched_road_segments_id', 'candidates'))
#
# # 修改 Transition，以使用新的 State 结构
# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
#
# class Memory(object):
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)
#
#     def push(self, trace, matched_road_segments_id, candidates, next_trace, next_matched_road_segments_id, next_candidates, action, reward):
#         state_namedtuple = State(trace, matched_road_segments_id, candidates)
#         next_state_namedtuple = State(next_trace, next_matched_road_segments_id, next_candidates)
#         self.memory.append(Transition(state_namedtuple, action, next_state_namedtuple, reward))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)


# 定义一个新的 namedtuple，包括 trace, matched_road_segments_id, 和 candidates
State = namedtuple('State', ('traces_encoding', 'matched_road_segments_encoding', 'trace', 'matched_road_segments_id', 'candidates'))

# 修改 Transition，以使用新的 State 结构
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, last_traces_encoding, last_matched_road_segments_encoding, trace, matched_road_segments_id, candidates,
             traces_encoding, matched_road_segments_encoding, next_trace, next_matched_road_segments_id, next_candidates,
             action, reward):
        state_namedtuple = State(last_traces_encoding, last_matched_road_segments_encoding, trace, matched_road_segments_id, candidates)
        next_state_namedtuple = State(traces_encoding, matched_road_segments_encoding, next_trace, next_matched_road_segments_id, next_candidates)
        self.memory.append(Transition(state_namedtuple, action, next_state_namedtuple, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
