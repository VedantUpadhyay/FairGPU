import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.vg_eval.dataset import load_mixed_trace
from experiments.vg_eval.metrics import (
    RequestResult,
    compute_metrics,
    print_summary_table,
)


def configure_urgency_mode(urgency_mode: str) -> None:
    """
    Configure in-process value-curve tau values for this eval run.
    """
    from faircpu import value_curves

    if urgency_mode == "high_contrast":
        value_curves.STEEP.tau = 5.0
        value_curves.SMOOTH.tau = 500.0
    else:
        value_curves.STEEP.tau = 25.0
        value_curves.SMOOTH.tau = 100.0
    print(
        f"Urgency mode: {urgency_mode} "
        f"(STEEP tau={value_curves.STEEP.tau}, "
        f"SMOOTH tau={value_curves.SMOOTH.tau})",
        flush=True,
    )


async def run_condition(condition: str, requests, model: str, output_dir: str):
    """
    Run one scheduling condition against the mixed trace.
    Returns list of RequestResult.
    """
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.sampling_params import SamplingParams

    print(f"\nStarting condition: {condition}")
    print(f"Model: {model}, " f"Requests: {len(requests)}")

    if condition == "chunked_prefill_fcfs":
        engine_args = AsyncEngineArgs(
            model=model,
            scheduling_policy="fcfs",
            enable_chunked_prefill=True,
            max_num_batched_tokens=512,
            max_model_len=2048,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
        )
    else:
        engine_args = AsyncEngineArgs(
            model=model,
            scheduling_policy=condition,
            max_model_len=2048,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
        )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    results = []
    experiment_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Offset arrival times to wall clock
    trace_start = requests[0].arrival_time
    wall_start = time.time()

    async def submit_request(req):
        # Wait until this request's arrival time
        target = wall_start + (req.arrival_time - trace_start)
        now = time.time()
        if target > now:
            await asyncio.sleep(target - now)

        # Truncate prompt to be safe.
        # opt-1.3b has 2048 context, reserve 256 for output.
        MAX_PROMPT_CHARS = 7000  # ~1750 tokens
        if len(req.prompt) > MAX_PROMPT_CHARS:
            req.prompt = req.prompt[:MAX_PROMPT_CHARS]

        # Set sampling parameters for generation.
        params = SamplingParams(
            max_tokens=256,
            temperature=0.0,
        )

        first_token_time = None
        completion_time = None
        n_tokens = 0
        actual_arrival = time.time()

        try:
            async for output in engine.generate(
                req.prompt,
                params,
                request_id=req.request_id,
                priority=req.priority,
            ):
                if first_token_time is None:
                    first_token_time = time.time()
                n_tokens = len(output.outputs[0].token_ids)
                if output.finished:
                    completion_time = time.time()
        except Exception as e:
            print(f"Request {req.request_id} failed: {e}", flush=True)
            if first_token_time is None:
                first_token_time = time.time()
            if completion_time is None:
                completion_time = time.time()

        results.append(
            RequestResult(
                request_id=req.request_id,
                sla_tier=req.sla_tier,
                arrival_time=actual_arrival,
                first_token_time=first_token_time or time.time(),
                completion_time=completion_time or time.time(),
                num_output_tokens=n_tokens,
            )
        )
        if len(results) % 100 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{len(results)}.json")
            with open(checkpoint_path, "w") as f:
                json.dump(
                    {
                        "condition": condition,
                        "n_completed": len(results),
                        "results": [
                            {
                                "request_id": r.request_id,
                                "sla_tier": r.sla_tier,
                                "ttft_ms": r.ttft_ms,
                                "e2e_ms": r.e2e_ms,
                            }
                            for r in results
                        ],
                    },
                    f,
                )
            print(f"Checkpoint saved: {len(results)} requests", flush=True)
        print(
            f"  [{condition}] {req.request_id} "
            f"TTFT={results[-1].ttft_ms:.0f}ms "
            f"E2E={results[-1].e2e_ms:.0f}ms",
            flush=True,
        )

    # Submit all requests concurrently
    await asyncio.gather(*[submit_request(r) for r in requests])

    pass  # engine cleanup handled automatically

    metrics = compute_metrics(results, condition, experiment_start)

    out_path = os.path.join(output_dir, f"{condition}_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "results": [
                    {
                        "request_id": r.request_id,
                        "sla_tier": r.sla_tier,
                        "ttft_ms": r.ttft_ms,
                        "e2e_ms": r.e2e_ms,
                        "num_output_tokens": r.num_output_tokens,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved -> {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="FairGPU VG Scheduler Eval")
    parser.add_argument(
        "--condition",
        choices=[
            "fcfs",
            "priority",
            "chunked_prefill_fcfs",
            "value_greedy",
            "value_greedy_aging",
        ],
        required=True,
    )
    parser.add_argument("--n_interactive", type=int, default=300)
    parser.add_argument("--n_batch", type=int, default=700)
    parser.add_argument("--n_requests", type=int, default=1000)
    parser.add_argument(
        "--urgency_mode",
        choices=["standard", "high_contrast"],
        default="standard",
    )
    parser.add_argument("--model", default="facebook/opt-1.3b")
    parser.add_argument("--output_dir", default="experiments/vg_eval/results/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_urgency_mode(args.urgency_mode)
    n_interactive = int(args.n_requests * 0.3)
    n_batch = int(args.n_requests * 0.7)

    # Load trace
    requests = load_mixed_trace(
        n_interactive=n_interactive,
        n_batch=n_batch,
        seed=args.seed,
        urgency_mode=args.urgency_mode,
    )

    # Run condition
    metrics = asyncio.run(
        run_condition(
            condition=args.condition,
            requests=requests,
            model=args.model,
            output_dir=args.output_dir,
        )
    )

    print_summary_table([metrics])


if __name__ == "__main__":
    main()
