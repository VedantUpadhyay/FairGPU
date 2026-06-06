from dataclasses import dataclass

import numpy as np

INTERACTIVE_STARVATION_MS = 2000.0
BATCH_STARVATION_MS = 120000.0


@dataclass
class RequestResult:
    request_id: str
    sla_tier: str
    arrival_time: float
    first_token_time: float
    completion_time: float
    num_output_tokens: int

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_time - self.arrival_time) * 1000

    @property
    def e2e_ms(self) -> float:
        return (self.completion_time - self.arrival_time) * 1000


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def compute_metrics(
    results: list[RequestResult], condition: str, experiment_start: float
) -> dict:
    """
    Compute all Paper 2 metrics from results list.
    Returns dict suitable for JSON serialization.
    """
    interactive = [r for r in results if r.sla_tier == "interactive"]
    batch = [r for r in results if r.sla_tier == "batch"]

    i_ttft = [r.ttft_ms for r in interactive]
    b_e2e = [r.e2e_ms for r in batch]

    i_starved = sum(1 for t in i_ttft if t > INTERACTIVE_STARVATION_MS)
    b_starved = sum(1 for t in b_e2e if t > BATCH_STARVATION_MS)
    starvation_rate = (i_starved + b_starved) / len(results) if results else 0.0

    # Throughput
    if results:
        total_duration = max(r.completion_time for r in results) - experiment_start
        throughput = len(results) / total_duration
    else:
        throughput = 0.0

    return {
        "condition": condition,
        "n_interactive": len(interactive),
        "n_batch": len(batch),
        "interactive_p50_ttft_ms": percentile(i_ttft, 50),
        "interactive_p95_ttft_ms": percentile(i_ttft, 95),
        "interactive_p99_ttft_ms": percentile(i_ttft, 99),
        "batch_p50_e2e_ms": percentile(b_e2e, 50),
        "batch_p99_e2e_ms": percentile(b_e2e, 99),
        "throughput_rps": round(throughput, 3),
        "starvation_rate": round(starvation_rate, 4),
        "interactive_starvation_rate": (
            i_starved / len(interactive) if interactive else 0.0
        ),
        "batch_starvation_rate": b_starved / len(batch) if batch else 0.0,
        "median_ttft_ms": round(percentile(i_ttft, 50), 2),
    }


def print_summary_table(all_metrics: list[dict]):
    """Print results table for Paper 2."""
    print("\n" + "=" * 72)
    print("VG SCHEDULER EVALUATION RESULTS")
    print("=" * 72)
    hdr = (
        f"{'Condition':<16} "
        f"{'P99 TTFT(ms)':<14} "
        f"{'P99 E2E(ms)':<13} "
        f"{'Throughput':<12} "
        f"{'Starve%':<8}"
    )
    print(hdr)
    print("-" * 72)
    for m in all_metrics:
        print(
            f"{m['condition']:<16} "
            f"{m['interactive_p99_ttft_ms']:<14.1f} "
            f"{m['batch_p99_e2e_ms']:<13.1f} "
            f"{m['throughput_rps']:<12.3f} "
            f"{m['starvation_rate'] * 100:<8.1f}%"
        )
    print("=" * 72)
