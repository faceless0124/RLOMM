from collections import namedtuple, deque
import random


State = namedtuple('State', (
    'traces_encoding',
    'matched_road_segments_encoding',
    'trace',
    'matched_road_segments_id',
    'candidates',
    'positive_samples',
    'negative_samples'
))

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
        
        state_namedtuple = State(last_traces_encoding, last_matched_road_segments_encoding,
                                 trace, matched_road_segments_id, candidates,
                                 last_positive_samples, last_negative_samples)

        next_state_namedtuple = State(traces_encoding, matched_road_segments_encoding,
                                      next_trace, next_matched_road_segments_id, next_candidates,
                                      next_positive_samples, next_negative_samples)

        self.memory.append(Transition(state_namedtuple, action, next_state_namedtuple, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
