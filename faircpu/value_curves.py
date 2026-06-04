import math
from dataclasses import dataclass


@dataclass
class TaskProfile:
    tau: float
    base: float = 1.0
    floor: float = 0.0


STEEP = TaskProfile(tau=25.0, base=1.0, floor=0.2)
SMOOTH = TaskProfile(tau=100.0, base=1.0, floor=0.0)


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


def value_greedy_key(request, current_time: float) -> float:
    """
    Sort key for VALUE_GREEDY queue ordering.
    Higher = schedule sooner.
    Used as: sorted(requests, key=lambda r:
             value_greedy_key(r, time.time()),
             reverse=True)
    """
    profile = assign_profile(request)
    wait_secs = max(0.0, current_time - request.arrival_time)
    return compute_decay_rate(profile, wait_secs)
