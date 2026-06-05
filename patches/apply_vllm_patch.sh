#!/bin/bash
# Apply FairGPU value_greedy patch to the
# pip-installed vllm package (not the submodule).
# Run after: pip install vllm

set -e

# Find pip-installed vllm location. A checked-out vllm/
# submodule can appear as a namespace package with
# __file__ = None, so search site-packages explicitly.
VLLM_PKG=$(python3 - <<'PYEOF'
import os
import site
import sys

search_roots = []
try:
    search_roots.extend(site.getsitepackages())
except AttributeError:
    pass
try:
    search_roots.append(site.getusersitepackages())
except AttributeError:
    pass
search_roots.extend(p for p in sys.path if "site-packages" in p)

seen = set()
for root in search_roots:
    if not root or root in seen:
        continue
    seen.add(root)
    candidate = os.path.join(root, "vllm", "__init__.py")
    if os.path.exists(candidate):
        print(os.path.join(root, "vllm"))
        raise SystemExit(0)

raise SystemExit(0)
PYEOF
)

if [ -n "$VLLM_PKG" ]; then
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
CONFIG_FILE="$CONFIG_FILE" python3 - <<'PYEOF'
import os
import re
path = os.environ["CONFIG_FILE"]
target = 'Literal["fcfs", "priority", "value_greedy", "value_greedy_aging"]'
with open(path) as f:
    c = f.read()

c, alias_count = re.subn(
    r'SchedulerPolicy\s*=\s*Literal\[[^\]]*\]',
    f'SchedulerPolicy = {target}',
    c,
)

literal_patterns = [
    r'Literal\[\s*([\"\'])fcfs\1\s*,\s*([\"\'])priority\2\s*\]',
    r'Literal\[\s*([\"\'])priority\1\s*,\s*([\"\'])fcfs\2\s*\]',
    r'Literal\[\s*([\"\'])fcfs\1\s*,\s*([\"\'])priority\2\s*,\s*([\"\'])value_greedy\3\s*\]',
    r'Literal\[\s*([\"\'])priority\1\s*,\s*([\"\'])fcfs\2\s*,\s*([\"\'])value_greedy\3\s*\]',
]
literal_count = 0
for pattern in literal_patterns:
    c, count = re.subn(pattern, target, c)
    literal_count += count

with open(path, 'w') as f:
    f.write(c)

if 'value_greedy_aging' not in c:
    raise SystemExit('scheduler.py patch failed: value_greedy_aging missing')
print(
    'scheduler.py patched with regex OK '
    f'(alias={alias_count}, literals={literal_count})'
)
PYEOF

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

# Verify pip-installed vllm patch.
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
else
echo "No pip-installed vllm package found; skipping pip patch"
fi

# Mirror the scheduler.py Literal patch into the local
# submodule when present so local verification with
# sys.path.insert(0, "vllm") observes the same policy.
LOCAL_CONFIG_FILE="./vllm/vllm/config/scheduler.py"
if [ -f "$LOCAL_CONFIG_FILE" ]; then
    echo "Patching local submodule $LOCAL_CONFIG_FILE"
    CONFIG_FILE="$LOCAL_CONFIG_FILE" python3 - <<'PYEOF'
import os
import re

path = os.environ["CONFIG_FILE"]
target = 'Literal["fcfs", "priority", "value_greedy", "value_greedy_aging"]'
with open(path) as f:
    c = f.read()

c, alias_count = re.subn(
    r'SchedulerPolicy\s*=\s*Literal\[[^\]]*\]',
    f'SchedulerPolicy = {target}',
    c,
)

literal_patterns = [
    r'Literal\[\s*([\"\'])fcfs\1\s*,\s*([\"\'])priority\2\s*\]',
    r'Literal\[\s*([\"\'])priority\1\s*,\s*([\"\'])fcfs\2\s*\]',
    r'Literal\[\s*([\"\'])fcfs\1\s*,\s*([\"\'])priority\2\s*,\s*([\"\'])value_greedy\3\s*\]',
    r'Literal\[\s*([\"\'])priority\1\s*,\s*([\"\'])fcfs\2\s*,\s*([\"\'])value_greedy\3\s*\]',
]
literal_count = 0
for pattern in literal_patterns:
    c, count = re.subn(pattern, target, c)
    literal_count += count

with open(path, "w") as f:
    f.write(c)

if "value_greedy_aging" not in c:
    raise SystemExit("local scheduler.py patch failed")
print(
    "local scheduler.py patched with regex OK "
    f"(alias={alias_count}, literals={literal_count})"
)
PYEOF
fi
