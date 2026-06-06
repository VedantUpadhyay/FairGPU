from faircpu.value_curves import (
    STEEP,
    SMOOTH,
    TaskProfile,
    assign_profile,
    compute_decay_rate,
    value_greedy_aging_key,
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


def test_aging_boost_fires_above_threshold():
    """Starving task must beat non-starving task."""
    now = time.time()
    r_starving = MockRequest(
        priority=0, num_prompt_tokens=2000, arrival_time=now - 60.0
    )
    r_fresh = MockRequest(priority=1, num_prompt_tokens=100, arrival_time=now - 1.0)

    key_starving = value_greedy_aging_key(
        r_starving, now, starvation_threshold_secs=30.0
    )
    key_fresh = value_greedy_aging_key(r_fresh, now, starvation_threshold_secs=30.0)

    assert key_starving > key_fresh, "Starving task must be served before fresh task"


def test_aging_preserves_decay_order_below_threshold():
    """Below threshold, decay ordering preserved."""
    now = time.time()
    r_urgent = MockRequest(priority=1, num_prompt_tokens=100, arrival_time=now - 5.0)
    r_patient = MockRequest(
        priority=0, num_prompt_tokens=2000, arrival_time=now - 5.0
    )

    key_urgent = value_greedy_aging_key(
        r_urgent, now, starvation_threshold_secs=30.0
    )
    key_patient = value_greedy_aging_key(
        r_patient, now, starvation_threshold_secs=30.0
    )

    assert key_urgent > key_patient, "Below threshold, urgency ordering preserved"


def test_aging_fifo_among_starving():
    """Among starving tasks, longest-waiting served first."""
    now = time.time()
    r_older = MockRequest(priority=0, num_prompt_tokens=500, arrival_time=now - 90.0)
    r_newer = MockRequest(priority=0, num_prompt_tokens=500, arrival_time=now - 40.0)

    key_older = value_greedy_aging_key(
        r_older, now, starvation_threshold_secs=30.0
    )
    key_newer = value_greedy_aging_key(
        r_newer, now, starvation_threshold_secs=30.0
    )

    assert key_older > key_newer, "Longer-waiting starving task served first"


def test_backpressure_stages_prefills():
    """Under memory pressure, new requests are staged."""
    from vllm.v1.core.sched.request_queue import ValueGreedyBackpressureRequestQueue

    q = ValueGreedyBackpressureRequestQueue(memory_threshold=0.85)

    q.set_memory_pressure(True)

    class FakePrefill:
        arrival_time = __import__("time").time() - 1
        priority = 1
        num_prompt_tokens = 100
        num_computed_tokens = 0
        request_id = "prefill-1"

    q.add_request(FakePrefill())
    assert len(q._queue) == 0
    assert len(q._staging) == 1

    q.set_memory_pressure(False)
    assert len(q._queue) == 1
    assert len(q._staging) == 0
    print("Backpressure staging test PASSED")
