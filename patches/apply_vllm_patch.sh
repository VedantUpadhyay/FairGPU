#!/bin/bash
# Apply FairGPU value_greedy patch to the
# pip-installed vllm package (not the submodule).
# Run after: pip install vllm

set -e

# Find pip-installed vllm location
VLLM_PKG=$(python3 -c \
  "import vllm; import os; \
   print(os.path.dirname(vllm.__file__))")

echo "Patching pip-installed vllm at: $VLLM_PKG"

# Patch 1: Add value_greedy to SchedulingPolicy enum
# and add ValueGreedyRequestQueue class
QUEUE_FILE="$VLLM_PKG/v1/core/sched/request_queue.py"
echo "Patching $QUEUE_FILE"

python3 - <<PYEOF
import re

with open("$QUEUE_FILE", "r") as f:
    content = f.read()

# Add import at top
if "from faircpu.value_curves import" not in content:
    content = content.replace(
        "import heapq",
        "import heapq\nimport time as _time\n"
        "import sys, os as _os\n"
        "sys.path.insert(0, '/workspace/FairGPU')\n"
        "from faircpu.value_curves import (\n"
        "    DEFAULT_STARVATION_THRESHOLD,\n"
        "    value_greedy_aging_key,\n"
        "    value_greedy_key,\n"
        ")\n",
    )
elif "value_greedy_aging_key" not in content:
    content = content.replace(
        "from faircpu.value_curves import value_greedy_key",
        "from faircpu.value_curves import (\n"
        "    DEFAULT_STARVATION_THRESHOLD,\n"
        "    value_greedy_aging_key,\n"
        "    value_greedy_key,\n"
        ")",
    )

# Add enum value after PRIORITY
if "VALUE_GREEDY" not in content:
    content = content.replace(
        '    PRIORITY = "priority"',
        '    PRIORITY = "priority"\n'
        '    VALUE_GREEDY = "value_greedy"'
    )
if "VALUE_GREEDY_AGING" not in content:
    content = content.replace(
        '    VALUE_GREEDY = "value_greedy"',
        '    VALUE_GREEDY = "value_greedy"\n'
        '    VALUE_GREEDY_AGING = "value_greedy_aging"'
    )

# Add ValueGreedyRequestQueue class before
# create_request_queue function
vgq_class = """

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

if "class ValueGreedyRequestQueue" not in content:
    content = content.replace(
        "def create_request_queue(",
        vgq_class + "def create_request_queue("
    )

vgaq_class = """

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

if "class ValueGreedyAgingRequestQueue" not in content:
    content = content.replace(
        "def create_request_queue(",
        vgaq_class + "def create_request_queue("
    )

# Add branch in create_request_queue
if "SchedulingPolicy.VALUE_GREEDY:" not in content:
    content = content.replace(
        '    elif policy == SchedulingPolicy.FCFS:',
        '    elif policy == SchedulingPolicy.VALUE_GREEDY:\n'
        '        return ValueGreedyRequestQueue()\n'
        '    elif policy == SchedulingPolicy.FCFS:'
    )
if "SchedulingPolicy.VALUE_GREEDY_AGING:" not in content:
    content = content.replace(
        '    elif policy == SchedulingPolicy.FCFS:',
        '    elif policy == SchedulingPolicy.VALUE_GREEDY_AGING:\n'
        '        return ValueGreedyAgingRequestQueue()\n'
        '    elif policy == SchedulingPolicy.FCFS:'
    )

with open("$QUEUE_FILE", "w") as f:
    f.write(content)

print("request_queue.py patched OK")
PYEOF

# Patch 2: Add value_greedy to SchedulerPolicy Literal
CONFIG_FILE="$VLLM_PKG/config/scheduler.py"
echo "Patching $CONFIG_FILE"
python3 -c "
import re
with open('$CONFIG_FILE') as f:
    c = f.read()
if 'value_greedy_aging' in c:
    print('scheduler.py already patched')
else:
    if 'Literal["fcfs", "priority", "value_greedy"]' in c:
        c = c.replace(
            'Literal["fcfs", "priority", "value_greedy"]',
            'Literal["fcfs", "priority", "value_greedy", '
            '"value_greedy_aging"]'
        )
    else:
        c = c.replace(
            'Literal["fcfs", "priority"]',
            'Literal["fcfs", "priority", "value_greedy", '
            '"value_greedy_aging"]'
        )
    with open('$CONFIG_FILE', 'w') as f:
        f.write(c)
    print('scheduler.py patched OK')
"

# Patch 3: Add value_greedy to arg_utils choices
ARG_FILE="$VLLM_PKG/../vllm/engine/arg_utils.py"
ARG_FILE2="$VLLM_PKG/engine/arg_utils.py"
for F in $ARG_FILE $ARG_FILE2; do
    if [ -f "$F" ]; then
        python3 -c "
with open('$F') as f:
    c = f.read()
if 'value_greedy_aging' in c:
    print('arg_utils.py already patched')
else:
    if '[\"fcfs\", \"priority\", \"value_greedy\"]' in c:
        c = c.replace(
            '[\"fcfs\", \"priority\", \"value_greedy\"]',
            '[\"fcfs\", \"priority\", \"value_greedy\", '
            '\"value_greedy_aging\"]'
        )
    else:
        c = c.replace(
            '[\"fcfs\", \"priority\"]',
            '[\"fcfs\", \"priority\", \"value_greedy\", '
            '\"value_greedy_aging\"]'
        )
    with open('$F', 'w') as f:
        f.write(c)
    print('arg_utils.py patched OK')
"
    fi
done

echo "All patches applied to pip vllm at $VLLM_PKG"

# Verify
python3 -c "
import sys
sys.path.insert(0, '/workspace/FairGPU')
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy, ValueGreedyAgingRequestQueue,
    ValueGreedyRequestQueue)
print('Verification OK:',
      SchedulingPolicy.VALUE_GREEDY,
      SchedulingPolicy.VALUE_GREEDY_AGING)
"
