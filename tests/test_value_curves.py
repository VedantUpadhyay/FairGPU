from faircpu.value_curves import (
    STEEP,
    SMOOTH,
    TaskProfile,
    assign_profile,
    compute_decay_rate,
    value_greedy_key,
)
import time


class MockRequest:
    def __init__(self, priority, num_prompt_tokens, arrival_time):
        self.priority = priority
        self.num_prompt_tokens = num_prompt_tokens
        self.arrival_time = arrival_time


def test_decay_decreases_over_time():
    r0 = compute_decay_rate(STEEP, 0.0)
    r10 = compute_decay_rate(STEEP, 10.0)
    r50 = compute_decay_rate(STEEP, 50.0)
    assert r0 > r10 > r50 >= 0.0


def test_steep_decays_faster_than_smooth():
    wait = 20.0
    assert compute_decay_rate(STEEP, wait) > compute_decay_rate(SMOOTH, wait)


def test_floor_stops_decay():
    rate = compute_decay_rate(STEEP, 10000.0)
    assert rate == 0.0


def test_assign_profile_priority_1():
    r = MockRequest(priority=1, num_prompt_tokens=1000, arrival_time=0.0)
    assert assign_profile(r) == STEEP


def test_assign_profile_priority_0():
    r = MockRequest(priority=0, num_prompt_tokens=1000, arrival_time=0.0)
    assert assign_profile(r) == SMOOTH


def test_assign_profile_short_prompt():
    r = MockRequest(priority=0, num_prompt_tokens=100, arrival_time=0.0)
    assert assign_profile(r) == STEEP


def test_value_greedy_key_ordering():
    now = time.time()
    r_urgent = MockRequest(
        priority=1, num_prompt_tokens=100, arrival_time=now - 30.0
    )
    r_batch = MockRequest(priority=0, num_prompt_tokens=2000, arrival_time=now - 5.0)
    assert value_greedy_key(r_urgent, now) > value_greedy_key(r_batch, now)


def test_zero_wait_time():
    now = time.time()
    r = MockRequest(priority=1, num_prompt_tokens=100, arrival_time=now)
    key = value_greedy_key(r, now)
    assert key >= 0.0
