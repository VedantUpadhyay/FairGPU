# AGENTS.md - vLLM Value-Curve Scheduler

## Project goal
Implement a Value-Greedy (VG) scheduler inside vLLM that
replaces the static FCFS sort key with a dynamic decay-rate
ordering. This is the Paper 2 baseline experiment.

## Repo context
- Base: vllm-project/vllm (latest main branch)
- Target file: vllm/v1/core/sched/scheduler.py
- Do not touch: KV cache allocation, BlockManager,
  chunked prefill logic - sort key only in Phase 1
- All changes must be backward compatible:
  VG is opt-in via --scheduling-policy value_greedy

## External reference scripts
- Some scripts referenced in future prompts may live outside this repo at:
  `/Users/vedantupadhyay/OneDrive/GRAD - FALL 23/UCSC/Capstone`
- Treat that path as read-only reference material unless explicitly instructed
  otherwise.
- When using scripts from that path, copy or adapt only the needed logic into
  this repo and keep changes compatible with vLLM's existing structure.

## Required prior-work review
Before writing any new scheduler code, read the prior capstone implementation
in this repository or in the external reference path above.

- Read `capstone/value_curves.py`, or the file where value curves are
  implemented under `capstone/`.
  - The decay formula, tau values, steep/smooth profiles, and
    `compute_decay_rate` function must be identical to the existing
    implementation.
  - Do not independently reimplement this logic. Import it if feasible; if an
    import is not feasible, copy it exactly and preserve behavior.
- Read `capstone/vaem_eval.py`.
  - The critical-task selection rule, `wait_time > alpha * tau`, is the same
    concept as the vLLM Value-Greedy sort key. Understand this connection
    before implementing scheduler integration.
- Read `capstone/full_alibaba_eval.py`.
  - Understand how the simulator assigns steep/smooth profiles to tasks. Apply
    the same profile-assignment logic to vLLM requests.

The goal is one consistent framework across both papers, not two independent
implementations.

## vLLM Integration
vLLM is tracked as a git submodule at `./vllm/`.
Our scheduler changes are NOT committed inside `vllm/`.
Instead they live as a patch file:
  `patches/vllm_value_greedy.patch`

To apply after cloning FairGPU fresh:
  `git submodule update --init`
  `bash patches/apply_vllm_patch.sh`

To update the patch after making new vLLM edits:
  `cd vllm && git diff > ../patches/vllm_value_greedy.patch`

## Coding standards
- Match vllm's existing code style exactly
- Every new function gets a docstring
- Every change gets a corresponding unit test
- No new dependencies

## Definition of done for each task
- Code runs without import errors
- Existing vllm tests pass (run: pytest tests/core/ -x)
- New test added for each new function
- Print/log output shows decay rates being computed
