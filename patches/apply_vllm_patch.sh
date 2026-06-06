#!/bin/bash
# Apply FairGPU scheduler patches to vLLM.
# Run from FairGPU root after the repo is cloned.

set -e

python3 - <<'PYEOF'
import os
import re
import site
import sys


FAIRGPU_ROOTS = [
    "/workspace/FairGPU",
    os.getcwd(),
]


REQUEST_QUEUE_IMPORT = """from faircpu.value_curves import (
    DEFAULT_STARVATION_THRESHOLD,
    value_greedy_aging_key,
    value_greedy_key,
)"""


VALUE_GREEDY_CLASS = """

class ValueGreedyRequestQueue(RequestQueue):
    \"\"\"Schedule by descending value decay rate.\"\"\"

    def __init__(self) -> None:
        self._queue: list = []

    def _sort(self) -> None:
        self._queue.sort(
            key=lambda r: value_greedy_key(
                r, _time.time()),
            reverse=True)

    def add_request(self, request) -> None:
        self._queue.append(request)
        self._sort()

    def pop_request(self):
        if not self._queue:
            raise IndexError(
                "pop from empty ValueGreedyQueue")
        self._sort()
        return self._queue.pop(0)

    def peek_request(self):
        if not self._queue:
            raise IndexError(
                "peek from empty ValueGreedyQueue")
        self._sort()
        return self._queue[0]

    def prepend_request(self, request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests) -> None:
        for r in requests:
            self._queue.append(r)
        self._sort()

    def remove_request(self, request) -> None:
        self._queue.remove(request)

    def remove_requests(self, requests) -> None:
        rs = set(requests)
        self._queue = [r for r in self._queue
                       if r not in rs]

    def add(self, request) -> None:
        self.add_request(request)

    def pop(self):
        return self.pop_request()

    def peek(self):
        if not self._queue:
            return None
        return self.peek_request()

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self):
        self._sort()
        return iter(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

"""


VALUE_GREEDY_AGING_CLASS = """

class ValueGreedyAgingRequestQueue(RequestQueue):
    \"\"\"
    Value-Greedy with bounded aging starvation prevention.
    Combines decay-rate ordering with hard wait ceiling.
    \"\"\"

    def __init__(self,
                 starvation_threshold: float =
                 DEFAULT_STARVATION_THRESHOLD) -> None:
        self._queue: list = []
        self._threshold = starvation_threshold

    def _sort(self) -> None:
        now = _time.time()
        self._queue.sort(
            key=lambda r: value_greedy_aging_key(
                r, now,
                starvation_threshold_secs=self._threshold),
            reverse=True)

    def add_request(self, request) -> None:
        self._queue.append(request)
        self._sort()

    def pop_request(self):
        if not self._queue:
            raise IndexError(
                "pop from empty ValueGreedyAgingQueue")
        self._sort()
        return self._queue.pop(0)

    def peek_request(self):
        if not self._queue:
            raise IndexError(
                "peek from empty ValueGreedyAgingQueue")
        self._sort()
        return self._queue[0]

    def prepend_request(self, request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests) -> None:
        for r in requests:
            self._queue.append(r)
        self._sort()

    def remove_request(self, request) -> None:
        self._queue.remove(request)

    def remove_requests(self, requests) -> None:
        rs = set(requests)
        self._queue = [r for r in self._queue
                       if r not in rs]

    def add(self, request) -> None:
        self.add_request(request)

    def pop(self):
        return self.pop_request()

    def peek(self):
        if not self._queue:
            return None
        return self.peek_request()

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self):
        self._sort()
        return iter(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

"""


VALUE_GREEDY_BACKPRESSURE_CLASS = """

class ValueGreedyBackpressureRequestQueue(RequestQueue):
    \"\"\"
    Value-Greedy with memory-aware admission control.

    When KV cache utilization exceeds threshold, new prefill
    requests are held in a staging queue and admitted when
    pressure drops.
    \"\"\"

    def __init__(self,
                 memory_threshold: float = 0.85,
                 starvation_threshold: float = 30.0,
                 boost: float = 1e9) -> None:
        self._queue: list = []
        self._staging: list = []
        self._memory_threshold = memory_threshold
        self._starvation_threshold = starvation_threshold
        self._boost = boost
        self._memory_pressure = False

    def set_memory_pressure(self, high_pressure: bool) -> None:
        \"\"\"Called by scheduler with current KV state.\"\"\"
        self._memory_pressure = high_pressure
        if high_pressure:
            held = [r for r in self._queue if self._is_prefill(r)]
            if held:
                held_set = set(held)
                self._queue = [r for r in self._queue
                               if r not in held_set]
                self._staging.extend(held)
        elif self._staging:
            self._queue.extend(self._staging)
            self._staging.clear()
            self._sort()

    def _is_prefill(self, request) -> bool:
        return (hasattr(request, 'num_computed_tokens')
                and request.num_computed_tokens == 0)

    def _sort(self) -> None:
        now = _time.time()
        self._queue.sort(
            key=lambda r: value_greedy_aging_key(
                r, now,
                starvation_threshold_secs=
                    self._starvation_threshold,
                boost=self._boost),
            reverse=True)

    def add_request(self, request) -> None:
        if self._memory_pressure and self._is_prefill(request):
            self._staging.append(request)
        else:
            self._queue.append(request)
            self._sort()

    def pop_request(self):
        if not self._queue:
            raise IndexError(
                "pop from empty BackpressureQueue")
        self._sort()
        return self._queue.pop(0)

    def peek_request(self):
        if not self._queue:
            raise IndexError(
                "peek from empty BackpressureQueue")
        self._sort()
        return self._queue[0]

    def prepend_request(self, request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests) -> None:
        for r in requests:
            self.add_request(r)

    def remove_request(self, request) -> None:
        try:
            self._queue.remove(request)
        except ValueError:
            self._staging.remove(request)

    def remove_requests(self, requests) -> None:
        rs = set(requests)
        self._queue = [r for r in self._queue
                       if r not in rs]
        self._staging = [r for r in self._staging
                         if r not in rs]

    def add(self, request) -> None:
        self.add_request(request)

    def pop(self):
        return self.pop_request()

    def peek(self):
        if not self._queue:
            return None
        return self.peek_request()

    def __len__(self) -> int:
        return len(self._queue) + len(self._staging)

    def __iter__(self):
        self._sort()
        return iter(self._queue + self._staging)

    def __bool__(self) -> bool:
        # Only the main queue is schedulable. Staged requests are
        # counted by __len__ but should not drive scheduler popping.
        return bool(self._queue)

"""


POLICY_LITERAL = (
    'Literal["fcfs", "priority", "value_greedy", '
    '"value_greedy_aging", "value_greedy_backpressure"]'
)
CHOICES_LITERAL = (
    '["fcfs", "priority", "value_greedy", '
    '"value_greedy_aging", "value_greedy_backpressure"]'
)


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def find_vllm_roots() -> list[str]:
    roots: list[str] = []
    search_roots: list[str] = []
    try:
        search_roots.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        search_roots.append(site.getusersitepackages())
    except AttributeError:
        pass
    search_roots.extend(p for p in sys.path if "site-packages" in p)

    for root in search_roots:
        candidate = os.path.join(root, "vllm", "__init__.py")
        if os.path.exists(candidate):
            roots.append(os.path.join(root, "vllm"))

    local_root = os.path.join(os.getcwd(), "vllm", "vllm")
    if os.path.exists(os.path.join(local_root, "__init__.py")):
        roots.append(local_root)

    deduped = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def patch_request_queue(root: str) -> None:
    path = os.path.join(root, "v1", "core", "sched", "request_queue.py")
    if not os.path.exists(path):
        return
    content = read(path)

    if "import time as _time" not in content:
        content = content.replace("import heapq\n", "import heapq\nimport time as _time\n")

    if "from faircpu.value_curves import" in content:
        content = re.sub(
            r"from faircpu\.value_curves import(?: \([^)]+\)| [^\n]+)",
            REQUEST_QUEUE_IMPORT,
            content,
            count=1,
            flags=re.S,
        )
    else:
        content = content.replace(
            "from vllm.v1.request import Request\n",
            REQUEST_QUEUE_IMPORT + "\n\nfrom vllm.v1.request import Request\n",
        )

    if "VALUE_GREEDY" not in content:
        content = content.replace(
            '    PRIORITY = "priority"',
            '    PRIORITY = "priority"\n'
            '    VALUE_GREEDY = "value_greedy"',
        )
    if "VALUE_GREEDY_AGING" not in content:
        content = content.replace(
            '    VALUE_GREEDY = "value_greedy"',
            '    VALUE_GREEDY = "value_greedy"\n'
            '    VALUE_GREEDY_AGING = "value_greedy_aging"',
        )
    if "VALUE_GREEDY_BACKPRESSURE" not in content:
        content = content.replace(
            '    VALUE_GREEDY_AGING = "value_greedy_aging"',
            '    VALUE_GREEDY_AGING = "value_greedy_aging"\n'
            '    VALUE_GREEDY_BACKPRESSURE = "value_greedy_backpressure"',
        )

    insert = ""
    if "class ValueGreedyRequestQueue" not in content:
        insert += VALUE_GREEDY_CLASS
    if "class ValueGreedyAgingRequestQueue" not in content:
        insert += VALUE_GREEDY_AGING_CLASS
    if "class ValueGreedyBackpressureRequestQueue" not in content:
        insert += VALUE_GREEDY_BACKPRESSURE_CLASS
    if insert:
        content = content.replace("def create_request_queue(", insert + "def create_request_queue(")

    if "SchedulingPolicy.VALUE_GREEDY:" not in content:
        content = content.replace(
            '    elif policy == SchedulingPolicy.FCFS:',
            '    elif policy == SchedulingPolicy.VALUE_GREEDY:\n'
            '        return ValueGreedyRequestQueue()\n'
            '    elif policy == SchedulingPolicy.FCFS:',
        )
    if "SchedulingPolicy.VALUE_GREEDY_AGING:" not in content:
        content = content.replace(
            '    elif policy == SchedulingPolicy.FCFS:',
            '    elif policy == SchedulingPolicy.VALUE_GREEDY_AGING:\n'
            '        return ValueGreedyAgingRequestQueue()\n'
            '    elif policy == SchedulingPolicy.FCFS:',
        )
    if "SchedulingPolicy.VALUE_GREEDY_BACKPRESSURE:" not in content:
        content = content.replace(
            '    elif policy == SchedulingPolicy.FCFS:',
            '    elif policy == SchedulingPolicy.VALUE_GREEDY_BACKPRESSURE:\n'
            '        return ValueGreedyBackpressureRequestQueue()\n'
            '    elif policy == SchedulingPolicy.FCFS:',
        )

    write(path, content)
    print(f"request_queue.py patched OK: {path}")


def patch_scheduler_config(root: str) -> None:
    path = os.path.join(root, "config", "scheduler.py")
    if not os.path.exists(path):
        return
    content = read(path)
    content, alias_count = re.subn(
        r"SchedulerPolicy\s*=\s*Literal\[[^\]]*\]",
        f"SchedulerPolicy = {POLICY_LITERAL}",
        content,
    )
    literal_patterns = [
        r"Literal\[\s*([\"\'])fcfs\1\s*,\s*([\"\'])priority\2\s*\]",
        r"Literal\[\s*([\"\'])priority\1\s*,\s*([\"\'])fcfs\2\s*\]",
        r"Literal\[\s*([\"\'])fcfs\1\s*,\s*([\"\'])priority\2\s*,\s*([\"\'])value_greedy\3\s*\]",
        r"Literal\[\s*([\"\'])fcfs\1\s*,\s*([\"\'])priority\2\s*,\s*([\"\'])value_greedy\3\s*,\s*([\"\'])value_greedy_aging\4\s*\]",
    ]
    literal_count = 0
    for pattern in literal_patterns:
        content, count = re.subn(pattern, POLICY_LITERAL, content)
        literal_count += count
    if "value_greedy_backpressure" not in content:
        raise SystemExit(f"scheduler.py patch failed: {path}")
    write(path, content)
    print(
        f"scheduler.py patched with regex OK: {path} "
        f"(alias={alias_count}, literals={literal_count})"
    )


def patch_arg_utils(root: str) -> None:
    candidates = [
        os.path.join(root, "engine", "arg_utils.py"),
        os.path.join(os.path.dirname(root), "vllm", "engine", "arg_utils.py"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        content = read(path)
        for old in [
            '["fcfs", "priority"]',
            '["fcfs", "priority", "value_greedy"]',
            '["fcfs", "priority", "value_greedy", "value_greedy_aging"]',
        ]:
            content = content.replace(old, CHOICES_LITERAL)
        if "value_greedy_backpressure" not in content:
            print(f"WARNING: arg_utils.py choices not found in {path}")
        write(path, content)
        print(f"arg_utils.py patched OK: {path}")


def patch_scheduler_loop(root: str) -> None:
    path = os.path.join(root, "v1", "core", "sched", "scheduler.py")
    if not os.path.exists(path):
        return
    content = read(path)
    marker = "Signal memory pressure to backpressure queues."
    if marker in content:
        print(f"scheduler.py memory-pressure hook already present: {path}")
        return
    needle = "        self.kv_cache_manager.new_step_starts()\n"
    hook = (
        "        self.kv_cache_manager.new_step_starts()\n\n"
        "        # Signal memory pressure to backpressure queues.\n"
        "        kv_cache_usage = self.kv_cache_manager.usage\n"
        "        for request_queue in (self.waiting, self.skipped_waiting):\n"
        "            if hasattr(request_queue, \"set_memory_pressure\"):\n"
        "                request_queue.set_memory_pressure(kv_cache_usage > 0.85)\n"
    )
    if needle not in content:
        raise SystemExit(f"Could not find KV step hook point in {path}")
    content = content.replace(needle, hook, 1)
    write(path, content)
    print(f"scheduler.py memory-pressure hook patched OK: {path}")


roots = find_vllm_roots()
if not roots:
    raise SystemExit("No vLLM package roots found")

for root in roots:
    print(f"Patching vLLM root: {root}")
    patch_request_queue(root)
    patch_scheduler_config(root)
    patch_arg_utils(root)
    patch_scheduler_loop(root)

print("All FairGPU vLLM patches applied.")
PYEOF

python3 - <<'PYEOF'
import os
import site
import sys

# Prefer the prebuilt pip-installed vLLM over the cloned
# source submodule. The source tree lacks vllm._C inside
# the official Docker image and must not win import order.
site_paths = []
try:
    site_paths.extend(site.getsitepackages())
except AttributeError:
    pass
try:
    site_paths.append(site.getusersitepackages())
except AttributeError:
    pass
site_paths.extend(p for p in sys.path if "site-packages" in p)
site_paths = [p for p in dict.fromkeys(site_paths) if p]

cwd = os.getcwd()
local_vllm_paths = {
    "vllm",
    os.path.join(cwd, "vllm"),
    "/workspace/FairGPU/vllm",
}
remaining = [
    p for p in sys.path
    if p not in site_paths and p not in local_vllm_paths
]
sys.path = site_paths + remaining

import vllm
from vllm.config.scheduler import SchedulerPolicy
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    ValueGreedyAgingRequestQueue,
    ValueGreedyBackpressureRequestQueue,
    ValueGreedyRequestQueue,
)

print("Verification vLLM path:", getattr(vllm, "__file__", None))
print("Verification OK:", SchedulerPolicy)
print(
    "Queues OK:",
    SchedulingPolicy.VALUE_GREEDY,
    SchedulingPolicy.VALUE_GREEDY_AGING,
    SchedulingPolicy.VALUE_GREEDY_BACKPRESSURE,
)
PYEOF
