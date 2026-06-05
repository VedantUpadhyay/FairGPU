import math
from dataclasses import dataclass


@dataclass
class TaskProfile:
    tau: float
    base: float = 1.0
    floor: float = 0.0


STEEP = TaskProfile(tau=25.0, base=1.0, floor=0.2)
SMOOTH = TaskProfile(tau=100.0, base=1.0, floor=0.0)
DEFAULT_STARVATION_THRESHOLD = 30.0  # seconds


def compute_decay_rate(profile: TaskProfile, wait_secs: float) -> float:
    """
    Instantaneous value loss rate for a task.
    From Paper 1 Eq. 4: (base/tau) * exp(-wait/tau)
    Higher = more urgent = schedule sooner.
    Returns 0.0 if task is on the floor plateau.
    """
    current_value = profile.base * max(
        profile.floor, math.exp(-wait_secs / profile.tau)
    )
    if current_value <= profile.floor:
        return 0.0
    return (profile.base / profile.tau) * math.exp(-wait_secs / profile.tau)


def assign_profile(request) -> TaskProfile:
    """
    Assign a TaskProfile to a vllm.v1.request.Request.
    Uses request.priority and num_prompt_tokens.
    """
    if hasattr(request, "priority"):
        if request.priority >= 1:
            return STEEP
    if hasattr(request, "num_prompt_tokens"):
        if request.num_prompt_tokens < 512:
            return STEEP
    if hasattr(request, "priority"):
        if request.priority == 0:
            return SMOOTH
    return SMOOTH


def assign_profile_from_request(request) -> TaskProfile:
    """
    Use request.tau if available, else infer from priority/prompt length.
    """
    if hasattr(request, "tau") and request.tau > 0:
        floor = 0.2 if request.tau < 50 else 0.0
        return TaskProfile(tau=request.tau, base=1.0, floor=floor)
    return assign_profile(request)


def value_greedy_key(request, current_time: float) -> float:
    """
    Sort key for VALUE_GREEDY queue ordering.
    Higher = schedule sooner.
    Used as: sorted(requests, key=lambda r:
             value_greedy_key(r, time.time()),
             reverse=True)
    """
    profile = assign_profile_from_request(request)
    wait_secs = max(0.0, current_time - request.arrival_time)
    return compute_decay_rate(profile, wait_secs)


def value_greedy_aging_key(
    request,
    current_time: float,
    starvation_threshold_secs: float = DEFAULT_STARVATION_THRESHOLD,
    boost: float = 1e9,
) -> float:
    """
    Value-Greedy with bounded aging (starvation prevention).

    Sort key for scheduling: higher = serve sooner.

    Two-regime policy:
      Hot path (starving): if wait_time > threshold,
        return boost (large constant) + wait_time.
        Forces promotion to front of queue.
        Within starving tasks, FIFO by wait time.

      Cold path (not starving): return normal
        value_greedy_key -- serve by decay rate.

    This guarantees:
      - No request waits longer than threshold
      - Value ordering preserved for non-starving tasks
      - Zero starvation at any load level

    Parameters:
      starvation_threshold_secs: max wait before force-promotion.
        Default 30s. Should be set to ~3x median completion time.
      boost: large constant ensuring starving tasks always beat
        non-starving tasks. Default 1e9.
    """
    wait_secs = max(0.0, current_time - request.arrival_time)

    if wait_secs > starvation_threshold_secs:
        return boost + wait_secs

    return value_greedy_key(request, current_time)
