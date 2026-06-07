"""
Scheduler-Aware GPU Capacity Planner for LLM Inference
=======================================================
Paper 3 simulation: shows how scheduler choice (FCFS vs VG+Aging)
affects the minimum GPU fleet size needed to meet a P99 TTFT SLA.

Based on M/G/c queueing theory with empirical service rates
measured from FairGPU experiments on Nautilus GPU cluster.

Usage:
    python capacity_sim.py

Outputs:
    - capacity_results.json  (raw data)
    - capacity_table.txt     (LaTeX-ready table)
    - scheduler_savings.txt  (key findings)
"""

import itertools
import csv
import json
import math
import os
import random
import urllib.request
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────
# Empirical service rates from FairGPU experiments
# (requests per second, measured on Nautilus A10 GPU)
# ─────────────────────────────────────────────────────────────

EMPIRICAL_MU = {
    # (model, scheduler) -> measured throughput (rps)
    # Source: FairGPU Nautilus experiments, June 2026
    ("opt-1.3b", "fcfs"):              7.472,
    ("opt-1.3b", "chunked_fcfs"):      8.228,
    ("opt-1.3b", "value_greedy"):      7.787,
    ("opt-1.3b", "value_greedy_aging"): 9.286,
    ("opt-6.7b", "fcfs"):              2.876,
    ("opt-6.7b", "value_greedy"):      2.760,
    ("opt-6.7b", "value_greedy_aging"): 1.517,  # saturated — not useful
}

# GPU hardware costs (USD per GPU-hour, cloud on-demand, mid-2026)
GPU_COST_PER_HOUR = {
    "A10":  1.20,   # NVIDIA A10 24GB — used in experiments
    "A100": 3.00,   # NVIDIA A100 80GB
    "H100": 3.93,   # NVIDIA H100 SXM5 (post AWS June 2025 cut)
}

# Throughput scaling relative to measured A10 baseline.
GPU_SCALING_FACTORS = {
    "A10": 1.0,
    "A100": 2.1,
    "H100": 3.2,
}

HOURS_PER_MONTH = 24 * 30
HOURS_PER_YEAR = 24 * 365
TRACE_WINDOW_MINUTES = 5
TRACE_WINDOW_HOURS = TRACE_WINDOW_MINUTES / 60
SAVINGS_WINDOW_LOW_RPS = EMPIRICAL_MU[("opt-1.3b", "fcfs")]
SAVINGS_WINDOW_HIGH_RPS = EMPIRICAL_MU[
    ("opt-1.3b", "value_greedy_aging")
]

# SLA target: P99 TTFT must be below this threshold (seconds)
SLA_P99_TTFT_SEC = 2.0   # 2000ms — matches FairGPU absolute threshold

# Coefficient of variation for service time (M/G/c input)
# Empirically estimated from prefill+decode time distribution
# Prefill is roughly deterministic given prompt length;
# decode varies with output length. CV ~0.7 is a reasonable estimate
# for mixed short/long output workloads.
CV_SERVICE_TIME = 0.7


# ─────────────────────────────────────────────────────────────
# M/G/c Queueing Model
# Approximation: Kingman's formula extended to M/G/c
# Uses the Cosmetatos (1976) approximation for P99 wait time
# ─────────────────────────────────────────────────────────────

def erlang_c(c: int, rho_total: float) -> float:
    """
    Erlang C formula: probability that an arriving customer
    must wait (all c servers are busy).

    Parameters
    ----------
    c        : number of servers (GPUs)
    rho_total: total offered load = lambda / mu (not per-server)

    Returns probability of waiting (0 to 1).
    """
    rho_per_server = rho_total / c
    if rho_per_server >= 1.0:
        return 1.0  # unstable — all arrivals wait

    # Compute P(0) denominator using Poisson and Erlang terms
    a = rho_total  # offered load = lambda / mu
    numerator = (a ** c) / math.factorial(c) * (1.0 / (1.0 - rho_per_server))

    denominator = sum(a**k / math.factorial(k) for k in range(c)) + numerator
    return numerator / denominator


def mgc_mean_wait(
    lam: float,
    mu: float,
    c: int,
    cv: float = CV_SERVICE_TIME,
) -> float:
    """
    Mean waiting time in M/G/c queue (seconds).
    Uses the Cosmetatos approximation:
        W_q ≈ C(c, rho) * (1 + cv^2) / 2 * (1 / (c*mu - lambda))

    Parameters
    ----------
    lam : arrival rate (requests/sec)
    mu  : per-server service rate (requests/sec per GPU)
    c   : number of servers (GPUs)
    cv  : coefficient of variation of service time

    Returns mean waiting time in queue (seconds).
    """
    rho_total = lam / mu          # total offered load
    rho_per = rho_total / c       # utilization per server

    if rho_per >= 1.0:
        return float('inf')       # unstable

    ec = erlang_c(c, rho_total)
    mean_service = 1.0 / mu       # mean service time per GPU

    # Kingman / Cosmetatos correction for general service times
    w_q = ec * (1.0 + cv**2) / 2.0 * mean_service / (1.0 - rho_per)
    return w_q


def mgc_p99_wait(
    lam: float,
    mu: float,
    c: int,
    cv: float = CV_SERVICE_TIME,
) -> float:
    """
    Approximate P99 waiting time in M/G/c queue.

    Uses the approximation:
        P(W > t) ≈ C(c,rho) * exp(-mu*(c - rho_total)*t)
    Solve for t at P(W>t) = 0.01:
        t_99 = -ln(0.01/C) / (mu*(c - rho_total))

    Returns P99 wait time in seconds (approximation).
    """
    rho_total = lam / mu
    rho_per = rho_total / c

    if rho_per >= 1.0:
        return float('inf')

    ec = erlang_c(c, rho_total)
    if ec <= 0.01:
        return 0.0   # virtually no waiting

    # Exponential tail approximation
    decay = mu * (c - rho_total)
    if decay <= 0:
        return float('inf')

    # Apply (1 + cv^2)/2 correction for general service times
    correction = (1.0 + cv**2) / 2.0
    t99 = -math.log(0.01 / (ec * correction)) / decay
    return max(0.0, t99)


# ─────────────────────────────────────────────────────────────
# Capacity planning function
# ─────────────────────────────────────────────────────────────

@dataclass
class CapacityResult:
    model: str
    scheduler: str
    lambda_rps: float
    mu_rps: float
    rho_single: float           # utilization with 1 GPU
    min_gpus: int               # minimum GPUs to meet SLA
    p99_wait_ms: float          # P99 wait at min_gpus
    utilization_pct: float      # GPU utilization at min_gpus
    gpu_type: str
    cost_per_hour: float        # total fleet cost/hour
    cost_per_month: float       # total fleet cost/month
    sla_met: bool


def find_min_gpus(
    lam: float,
    mu: float,
    sla_sec: float = SLA_P99_TTFT_SEC,
    max_gpus: int = 20,
) -> tuple[int, float]:
    """
    Find minimum number of GPUs (c) such that P99 TTFT <= sla_sec.
    Returns (min_gpus, p99_wait_sec).
    """
    for c in range(1, max_gpus + 1):
        rho_per = (lam / mu) / c
        if rho_per >= 1.0:
            continue   # unstable with c GPUs, try more
        p99 = mgc_p99_wait(lam, mu, c)
        if p99 <= sla_sec:
            return c, p99
    return max_gpus, float('inf')


def plan_capacity(
    model: str,
    scheduler: str,
    lambda_rps: float,
    gpu_type: str = "A10",
    sla_sec: float = SLA_P99_TTFT_SEC,
) -> Optional[CapacityResult]:
    """
    Compute minimum GPU fleet for given model+scheduler+traffic.
    """
    base_mu = EMPIRICAL_MU.get((model, scheduler))
    if base_mu is None:
        return None
    mu = base_mu * GPU_SCALING_FACTORS.get(gpu_type, 1.0)

    rho_single = lambda_rps / mu
    min_gpus, p99_wait = find_min_gpus(lambda_rps, mu, sla_sec)

    sla_met = p99_wait <= sla_sec and p99_wait < float('inf')
    utilization = (lambda_rps / mu) / min_gpus * 100.0

    gpu_cost = GPU_COST_PER_HOUR.get(gpu_type, 1.20)
    cost_hour = min_gpus * gpu_cost
    cost_month = cost_hour * HOURS_PER_MONTH

    return CapacityResult(
        model=model,
        scheduler=scheduler,
        lambda_rps=lambda_rps,
        mu_rps=mu,
        rho_single=rho_single,
        min_gpus=min_gpus,
        p99_wait_ms=p99_wait * 1000.0,
        utilization_pct=utilization,
        gpu_type=gpu_type,
        cost_per_hour=cost_hour,
        cost_per_month=cost_month,
        sla_met=sla_met,
    )


# ─────────────────────────────────────────────────────────────
# Multi-GPU and trace analyses
# ─────────────────────────────────────────────────────────────

def analyze_multi_gpu(
    lambda_rps: float = 10.0,
    midpoint_lambda_rps: float = 7.0,
) -> dict[str, Any]:
    """
    Compare FCFS and VG+Aging capacity across GPU types.
    """
    capacity_rows = []
    annual_rows = []

    for gpu_type in ("A10", "A100", "H100"):
        fcfs = plan_capacity("opt-1.3b", "fcfs", lambda_rps, gpu_type)
        vga = plan_capacity(
            "opt-1.3b",
            "value_greedy_aging",
            lambda_rps,
            gpu_type,
        )
        if fcfs is None or vga is None:
            continue

        saved = max(0, fcfs.min_gpus - vga.min_gpus)
        monthly_saved = saved * GPU_COST_PER_HOUR[gpu_type] * HOURS_PER_MONTH
        capacity_rows.append({
            "gpu": gpu_type,
            "lambda_rps": lambda_rps,
            "fcfs_gpus": fcfs.min_gpus,
            "vg_aging_gpus": vga.min_gpus,
            "gpus_saved": saved,
            "monthly_savings_usd": monthly_saved,
            "fcfs_mu_rps": fcfs.mu_rps,
            "vg_aging_mu_rps": vga.mu_rps,
        })

        fcfs_mid = plan_capacity(
            "opt-1.3b",
            "fcfs",
            midpoint_lambda_rps,
            gpu_type,
        )
        vga_mid = plan_capacity(
            "opt-1.3b",
            "value_greedy_aging",
            midpoint_lambda_rps,
            gpu_type,
        )
        if fcfs_mid is None or vga_mid is None:
            continue
        annual_saved = max(
            0,
            fcfs_mid.min_gpus - vga_mid.min_gpus,
        ) * GPU_COST_PER_HOUR[gpu_type] * HOURS_PER_YEAR
        annual_rows.append({
            "gpu": gpu_type,
            "gpu_price_per_hour_usd": GPU_COST_PER_HOUR[gpu_type],
            "lambda_rps": midpoint_lambda_rps,
            "annual_savings_usd": annual_saved,
        })

    return {
        "scaling_factors": GPU_SCALING_FACTORS,
        "lambda_10_capacity": capacity_rows,
        "lambda_7_annual_savings": annual_rows,
    }


def print_multi_gpu_tables(multi_gpu: dict[str, Any]) -> None:
    """
    Print the requested multi-GPU capacity and savings tables.
    """
    print()
    print("─" * 72)
    print("Multi-GPU comparison at λ=10 rps")
    print("─" * 72)
    print(f"{'GPU':<8} {'FCFS GPUs':<12} {'VG+Aging GPUs':<16} "
          f"{'Saved':<8} {'$/month saved':<15}")
    print("-" * 62)
    for row in multi_gpu["lambda_10_capacity"]:
        print(f"{row['gpu']:<8} {row['fcfs_gpus']:<12} "
              f"{row['vg_aging_gpus']:<16} {row['gpus_saved']:<8} "
              f"${row['monthly_savings_usd']:,.0f}")

    print()
    print("─" * 72)
    print("Annual savings at λ=7 rps")
    print("─" * 72)
    print(f"{'GPU':<8} {'GPU price/hr':<16} {'Annual savings':<16}")
    print("-" * 42)
    for row in multi_gpu["lambda_7_annual_savings"]:
        print(f"{row['gpu']:<8} "
              f"${row['gpu_price_per_hour_usd']:<15.2f} "
              f"${row['annual_savings_usd']:,.0f}")


def parse_timestamp(value: Any) -> Optional[datetime]:
    """
    Parse common timestamp formats from public trace rows.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        stamp = float(value)
        if stamp > 1_000_000_000_000:
            stamp /= 1000.0
        if stamp > 1_000_000_000:
            return datetime.fromtimestamp(stamp, tz=timezone.utc)
        return None

    text = str(value).strip()
    if not text:
        return None
    try:
        return parse_timestamp(float(text))
    except ValueError:
        pass

    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def find_timestamp_in_record(record: dict[str, Any]) -> Optional[datetime]:
    """
    Find a timestamp-like field in a trace record.
    """
    preferred = (
        "timestamp",
        "tstamp",
        "created_at",
        "create_time",
        "arrival_time",
        "time",
        "date",
    )
    lowered = {str(k).lower(): v for k, v in record.items()}
    for key in preferred:
        if key in lowered:
            parsed = parse_timestamp(lowered[key])
            if parsed is not None:
                return parsed
    for key, value in record.items():
        key_text = str(key).lower()
        if "time" in key_text or "date" in key_text:
            parsed = parse_timestamp(value)
            if parsed is not None:
                return parsed
    return None


def timestamps_to_arrival_rates(
    timestamps: list[datetime],
    window_minutes: int = TRACE_WINDOW_MINUTES,
) -> list[float]:
    """
    Convert request timestamps into per-window arrival rates in rps.
    """
    if not timestamps:
        return []
    window_secs = window_minutes * 60
    buckets = Counter(
        int(ts.timestamp()) // window_secs
        for ts in timestamps
    )
    start = min(buckets)
    end = max(buckets)
    return [
        buckets.get(bucket, 0) / window_secs
        for bucket in range(start, end + 1)
    ]


def generate_synthetic_trace(
    trace_name: str,
    days: int = 7,
    window_minutes: int = TRACE_WINDOW_MINUTES,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generate a diurnal synthetic LLM traffic trace.
    """
    rng = random.Random(seed)
    n_windows = days * 24 * 60 // window_minutes
    rates = []
    for idx in range(n_windows):
        hour = (idx * window_minutes / 60.0) % 24.0
        diurnal = 7.0 + 5.0 * math.cos(2.0 * math.pi * (hour - 14.0) / 24.0)
        trough_correction = -0.75 * math.exp(-((hour - 4.0) ** 2) / 6.0)
        lam = diurnal + trough_correction + rng.gauss(0.0, 0.65)
        rates.append(max(0.0, min(14.0, lam)))
    return {
        "trace_name": trace_name,
        "source": "synthetic_fallback",
        "window_minutes": window_minutes,
        "arrival_rates_rps": rates,
    }


def load_lmsys_trace() -> dict[str, Any]:
    """
    Load LMSYS Chatbot Arena timestamps and convert them to rps windows.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "lmsys/chatbot_arena_conversations",
            split="train",
            streaming=True,
        )
        timestamps = []
        for item in itertools.islice(dataset, 50_000):
            parsed = find_timestamp_in_record(item)
            if parsed is not None:
                timestamps.append(parsed)
        if len(timestamps) < 10:
            raise ValueError("no usable timestamp field found")
        return {
            "trace_name": "lmsys_chatbot_arena",
            "source": "real_lmsys_chatbot_arena_conversations",
            "window_minutes": TRACE_WINDOW_MINUTES,
            "arrival_rates_rps": timestamps_to_arrival_rates(timestamps),
            "n_timestamps": len(timestamps),
        }
    except Exception as exc:
        trace = generate_synthetic_trace(
            "lmsys_chatbot_arena",
            seed=101,
        )
        trace["fallback_reason"] = str(exc)
        return trace


def load_azure_trace() -> dict[str, Any]:
    """
    Load Azure LLM inference trace if a public CSV path is available.
    """
    urls = (
        "https://raw.githubusercontent.com/Azure/"
        "AzurePublicDataset/main/AzureLLMInferenceTrace_conv.csv",
        "https://raw.githubusercontent.com/Azure/"
        "AzurePublicDataset/master/AzureLLMInferenceTrace_conv.csv",
        "https://raw.githubusercontent.com/Azure/"
        "AzurePublicDataset/main/data/AzureLLMInferenceTrace_conv.csv",
        "https://raw.githubusercontent.com/Azure/"
        "AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv",
    )
    errors = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                text = response.read(20_000_000).decode("utf-8", "replace")
            rows = csv.DictReader(text.splitlines())
            timestamps = []
            counts_by_minute = []
            for row in itertools.islice(rows, 200_000):
                parsed = find_timestamp_in_record(row)
                if parsed is not None:
                    timestamps.append(parsed)
                    continue
                count_value = None
                for key, value in row.items():
                    key_text = str(key).lower()
                    if "count" in key_text or "request" in key_text:
                        try:
                            count_value = float(value)
                            break
                        except (TypeError, ValueError):
                            continue
                if count_value is not None:
                    counts_by_minute.append(count_value / 60.0)
            if timestamps:
                rates = timestamps_to_arrival_rates(timestamps)
            else:
                rates = counts_by_minute
            if not rates:
                raise ValueError("no timestamp or count columns found")
            return {
                "trace_name": "azure_llm_inference",
                "source": f"real_azure_public_dataset:{url}",
                "window_minutes": TRACE_WINDOW_MINUTES,
                "arrival_rates_rps": rates,
                "n_rows": len(timestamps) + len(counts_by_minute),
            }
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    trace = generate_synthetic_trace(
        "azure_llm_inference",
        seed=202,
    )
    trace["fallback_reason"] = " | ".join(errors)
    return trace


def analyze_trace(trace: dict[str, Any], gpu_type: str = "A10") -> dict[str, Any]:
    """
    Compute savings-window coverage and expected GPU savings for a trace.
    """
    rates = trace["arrival_rates_rps"]
    in_window = [
        lam for lam in rates
        if SAVINGS_WINDOW_LOW_RPS < lam < SAVINGS_WINDOW_HIGH_RPS
    ]
    trace_hours = len(rates) * TRACE_WINDOW_HOURS
    raw_savings = 0.0
    for lam in rates:
        fcfs = plan_capacity("opt-1.3b", "fcfs", lam, gpu_type)
        vga = plan_capacity(
            "opt-1.3b",
            "value_greedy_aging",
            lam,
            gpu_type,
        )
        if fcfs is None or vga is None:
            continue
        gpu_saved = max(0, fcfs.min_gpus - vga.min_gpus)
        raw_savings += (
            gpu_saved * GPU_COST_PER_HOUR[gpu_type] * TRACE_WINDOW_HOURS
        )

    monthly_savings = (
        raw_savings * (HOURS_PER_MONTH / trace_hours)
        if trace_hours else 0.0
    )
    return {
        "trace_name": trace["trace_name"],
        "source": trace["source"],
        "fallback_reason": trace.get("fallback_reason"),
        "n_windows": len(rates),
        "window_minutes": TRACE_WINDOW_MINUTES,
        "gpu_type": gpu_type,
        "savings_window_rps": [
            SAVINGS_WINDOW_LOW_RPS,
            SAVINGS_WINDOW_HIGH_RPS,
        ],
        "fraction_in_savings_window": (
            len(in_window) / len(rates) if rates else 0.0
        ),
        "hours_in_savings_window": len(in_window) * TRACE_WINDOW_HOURS,
        "expected_monthly_savings_usd": monthly_savings,
        "expected_annual_savings_usd": monthly_savings * 12.0,
        "mean_lambda_rps": sum(rates) / len(rates) if rates else 0.0,
        "max_lambda_rps": max(rates) if rates else 0.0,
    }


def save_trace_histogram(
    traces: list[dict[str, Any]],
    output_path: str,
) -> None:
    """
    Save a histogram of trace arrival rates with the savings window shaded.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for trace in traces:
        plt.hist(
            trace["arrival_rates_rps"],
            bins=40,
            alpha=0.45,
            label=trace["trace_name"],
        )
    plt.axvspan(
        SAVINGS_WINDOW_LOW_RPS,
        SAVINGS_WINDOW_HIGH_RPS,
        color="green",
        alpha=0.18,
        label="VG+Aging savings window",
    )
    plt.xlabel("Arrival rate λ (requests/sec)")
    plt.ylabel("5-minute windows")
    plt.title("LLM Traffic Arrival Rates vs Capacity Savings Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_trace_analysis(results_dir: Path) -> dict[str, Any]:
    """
    Load real traces when possible, otherwise use synthetic fallback traces.
    """
    traces = [load_lmsys_trace(), load_azure_trace()]
    analyses = [analyze_trace(trace) for trace in traces]
    save_trace_histogram(
        traces,
        str(results_dir / "capacity_trace_histogram.png"),
    )
    return {
        "metadata": {
            "savings_window_low_rps": SAVINGS_WINDOW_LOW_RPS,
            "savings_window_high_rps": SAVINGS_WINDOW_HIGH_RPS,
            "gpu_type": "A10",
            "gpu_price_per_hour_usd": GPU_COST_PER_HOUR["A10"],
        },
        "traces": analyses,
    }


# ─────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────

def run_analysis():
    results = []
    savings = []

    models = ["opt-1.3b", "opt-6.7b"]
    schedulers_1b = ["fcfs", "chunked_fcfs",
                     "value_greedy", "value_greedy_aging"]
    schedulers_67b = ["fcfs", "value_greedy"]

    # Sweep arrival rates
    lambdas_1b = [2, 4, 6, 7, 8, 9, 10, 12, 15]
    lambdas_67b = [1, 2, 3, 4, 5]

    gpu_type = "A10"

    print("=" * 72)
    print("SCHEDULER-AWARE GPU CAPACITY PLANNER — FairGPU Paper 3")
    print("=" * 72)
    print(f"SLA target: P99 TTFT <= {SLA_P99_TTFT_SEC*1000:.0f}ms")
    print(f"GPU type: {gpu_type} @ ${GPU_COST_PER_HOUR[gpu_type]:.2f}/hr")
    print()

    # ── opt-1.3b analysis ──────────────────────────────────────
    print("─" * 72)
    print("opt-1.3b: Minimum GPUs to Meet P99 TTFT SLA")
    print("─" * 72)
    print(f"{'λ (rps)':<10} {'FCFS GPUs':<12} {'VG+Aging GPUs':<16} "
          f"{'GPU Saved':<12} {'$/month saved':<15}")
    print("-" * 65)

    for lam in lambdas_1b:
        r_fcfs = plan_capacity("opt-1.3b", "fcfs", lam, gpu_type)
        r_vga  = plan_capacity("opt-1.3b", "value_greedy_aging",
                               lam, gpu_type)
        if r_fcfs and r_vga:
            results.append(asdict(r_fcfs))
            results.append(asdict(r_vga))
            saved_gpus = r_fcfs.min_gpus - r_vga.min_gpus
            saved_monthly = (r_fcfs.cost_per_month
                             - r_vga.cost_per_month)
            fcfs_label = (f"{r_fcfs.min_gpus}"
                          + ("*" if not r_fcfs.sla_met else ""))
            vga_label  = (f"{r_vga.min_gpus}"
                          + ("*" if not r_vga.sla_met else ""))
            savings_label = (f"-{saved_gpus}"
                             if saved_gpus > 0 else "0")
            money_label = (f"-${saved_monthly:,.0f}"
                           if saved_monthly > 0 else "$0")
            print(f"{lam:<10} {fcfs_label:<12} {vga_label:<16} "
                  f"{savings_label:<12} {money_label:<15}")
            if saved_gpus > 0:
                savings.append({
                    "lambda_rps": lam,
                    "model": "opt-1.3b",
                    "fcfs_gpus": r_fcfs.min_gpus,
                    "vga_gpus": r_vga.min_gpus,
                    "gpus_saved": saved_gpus,
                    "monthly_savings_usd": saved_monthly,
                    "annual_savings_usd": saved_monthly * 12,
                })

    print("* = SLA not met even at max GPUs tested")
    print()

    # ── opt-1.3b full comparison table ────────────────────────
    print("─" * 72)
    print("opt-1.3b at λ=10 rps: All Schedulers Compared")
    print("─" * 72)
    print(f"{'Scheduler':<22} {'μ (rps)':<10} {'Min GPUs':<10} "
          f"{'P99 TTFT':<12} {'Util%':<8} {'$/month':<10}")
    print("-" * 72)

    for sched in schedulers_1b:
        r = plan_capacity("opt-1.3b", sched, 10.0, gpu_type)
        if r:
            results.append(asdict(r))
            sla_mark = "✓" if r.sla_met else "✗"
            print(f"{sched:<22} {r.mu_rps:<10.3f} "
                  f"{r.min_gpus}{sla_mark:<9} "
                  f"{r.p99_wait_ms:<12.1f} "
                  f"{r.utilization_pct:<8.1f} "
                  f"${r.cost_per_month:,.0f}")

    print()

    # ── opt-6.7b analysis ──────────────────────────────────────
    print("─" * 72)
    print("opt-6.7b at various λ: Saturation Boundary")
    print("─" * 72)
    print(f"{'λ (rps)':<10} {'FCFS GPUs':<12} {'ρ/GPU':<10} "
          f"{'P99 TTFT':<14} {'SLA met?':<10}")
    print("-" * 56)

    for lam in lambdas_67b:
        r = plan_capacity("opt-6.7b", "fcfs", lam, gpu_type)
        if r:
            results.append(asdict(r))
            rho_label = f"{r.rho_single:.2f}"
            p99_label = (f"{r.p99_wait_ms:.0f}ms"
                         if r.p99_wait_ms < 1e6
                         else "∞")
            sla_label = "✓" if r.sla_met else "✗ saturated"
            print(f"{lam:<10} {r.min_gpus:<12} {rho_label:<10} "
                  f"{p99_label:<14} {sla_label}")

    print()

    # ── Scheduler savings summary ──────────────────────────────
    print("=" * 72)
    print("SCHEDULER SAVINGS SUMMARY (opt-1.3b, A10 GPU)")
    print("=" * 72)
    if savings:
        for s in savings:
            print(f"  λ={s['lambda_rps']} rps: "
                  f"VG+Aging saves {s['gpus_saved']} GPU "
                  f"vs FCFS → "
                  f"${s['monthly_savings_usd']:,.0f}/month, "
                  f"${s['annual_savings_usd']:,.0f}/year")
    else:
        print("  No GPU savings at tested traffic levels.")
        print("  (Both schedulers need same fleet size "
              "at these arrival rates)")

    # ── Key finding ───────────────────────────────────────────
    print()
    print("─" * 72)
    print("KEY FINDING")
    print("─" * 72)
    mu_fcfs = EMPIRICAL_MU[("opt-1.3b", "fcfs")]
    mu_vga  = EMPIRICAL_MU[("opt-1.3b", "value_greedy_aging")]
    improvement = (mu_vga - mu_fcfs) / mu_fcfs * 100
    print(f"  VG+Aging effective throughput: {mu_vga:.3f} rps")
    print(f"  FCFS effective throughput:     {mu_fcfs:.3f} rps")
    print(f"  Improvement: +{improvement:.1f}%")
    print()
    print(f"  The single-GPU operating range extends from:")
    print(f"    FCFS:     λ < {mu_fcfs:.1f} rps")
    print(f"    VG+Aging: λ < {mu_vga:.1f} rps")
    print()
    window = mu_vga - mu_fcfs
    cost_per_gpu_month = GPU_COST_PER_HOUR["A10"] * 24 * 30
    print(f"  Traffic window where VG+Aging saves 1 GPU:")
    print(f"    {mu_fcfs:.1f} < λ < {mu_vga:.1f} rps "
          f"({window:.2f} rps wide)")
    print(f"  Cost of 1 A10 GPU: "
          f"${cost_per_gpu_month:,.0f}/month, "
          f"${cost_per_gpu_month*12:,.0f}/year")
    print()
    print(f"  Operators running in this window can defer hardware")
    print(f"  procurement by deploying VG+Aging scheduler instead.")

    # ── LaTeX table output ────────────────────────────────────
    latex = generate_latex_table(lambdas_1b)
    print()
    print("─" * 72)
    print("LaTeX TABLE (paste into paper)")
    print("─" * 72)
    print(latex)

    # ── Multi-GPU comparison (A10, A100, H100) ─────────────────
    multi_gpu = analyze_multi_gpu()
    print_multi_gpu_tables(multi_gpu)

    # ── Trace analysis with synthetic fallback ────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print()
    print("=" * 72)
    print("TRACE-BASED CAPACITY ANALYSIS")
    print("=" * 72)
    trace_analysis = run_trace_analysis(results_dir)
    for ta in trace_analysis["traces"]:
        src = ("SYNTHETIC" if ta.get("fallback_reason")
               else ta["source"])
        print(f"\n  Trace: {ta['trace_name']}  ({src})")
        print(f"    Windows:   {ta['n_windows']}")
        print(f"    Mean λ:    {ta['mean_lambda_rps']:.2f} rps")
        print(f"    Max  λ:    {ta['max_lambda_rps']:.2f} rps")
        pct = ta['fraction_in_savings_window'] * 100
        print(f"    In savings window: {pct:.1f}%")
        print(f"    Expected monthly savings: "
              f"${ta['expected_monthly_savings_usd']:,.0f}")
        print(f"    Expected annual  savings: "
              f"${ta['expected_annual_savings_usd']:,.0f}")
        if ta.get("fallback_reason"):
            print(f"    Fallback reason: {ta['fallback_reason'][:120]}")

    print()
    print(f"  Histogram → results/capacity_trace_histogram.png")

    # ── Save results ──────────────────────────────────────────
    output = {
        "metadata": {
            "sla_p99_ttft_ms": SLA_P99_TTFT_SEC * 1000,
            "gpu_type": "A10",
            "gpu_cost_per_hour": GPU_COST_PER_HOUR["A10"],
            "empirical_mu": {
                str(k): v for k, v in EMPIRICAL_MU.items()
            },
        },
        "results": results,
        "savings": savings,
        "multi_gpu": multi_gpu,
        "trace_analysis": trace_analysis,
    }

    json_path = results_dir / "capacity_results.json"
    tex_path = results_dir / "capacity_table.tex"
    summary_path = results_dir / "scheduler_savings.txt"

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    with open(tex_path, "w") as f:
        f.write(latex)

    with open(summary_path, "w") as f:
        f.write("Scheduler-Aware GPU Capacity Savings\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"SLA: P99 TTFT <= {SLA_P99_TTFT_SEC*1000:.0f}ms\n")
        f.write(f"GPU: A10 @ ${GPU_COST_PER_HOUR['A10']:.2f}/hr\n\n")
        f.write("Per-lambda savings (opt-1.3b, A10):\n")
        for s in savings:
            f.write(f"  λ={s['lambda_rps']} rps: "
                    f"{s['gpus_saved']} GPU saved → "
                    f"${s['monthly_savings_usd']:,.0f}/month\n")
        f.write("\nMulti-GPU scaling (λ=10 rps):\n")
        for row in multi_gpu["lambda_10_capacity"]:
            f.write(f"  {row['gpu']}: FCFS {row['fcfs_gpus']} → "
                    f"VG+Aging {row['vg_aging_gpus']} "
                    f"(save {row['gpus_saved']} GPU, "
                    f"${row['monthly_savings_usd']:,.0f}/month)\n")
        f.write("\nAnnual savings (λ=7 rps):\n")
        for row in multi_gpu["lambda_7_annual_savings"]:
            f.write(f"  {row['gpu']}: "
                    f"${row['annual_savings_usd']:,.0f}/year\n")
        f.write("\nTrace-driven estimates:\n")
        for ta in trace_analysis["traces"]:
            f.write(f"  {ta['trace_name']}: "
                    f"${ta['expected_annual_savings_usd']:,.0f}/year "
                    f"({ta['fraction_in_savings_window']*100:.0f}% "
                    f"windows in savings range)\n")

    print()
    print("─" * 72)
    print("Saved outputs:")
    print(f"  {json_path}")
    print(f"  {tex_path}")
    print(f"  {summary_path}")
    print(f"  results/capacity_trace_histogram.png")
    print("─" * 72)
    return output


def generate_latex_table(lambdas: list) -> str:
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Scheduler-aware GPU capacity planning for "
        r"\texttt{opt-1.3b} on A10 GPU (\$1.20/hr). "
        r"Minimum GPU count to meet P99 TTFT $\leq$ 2{,}000\,ms SLA. "
        r"VG+Aging extends the single-GPU operating range "
        r"from $\lambda < 7.5$\,rps to $\lambda < 9.3$\,rps, "
        r"saving one GPU for traffic in that window.}"
    )
    lines.append(r"\label{tab:capacity}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"$\lambda$ (rps) & \fcfs\ GPUs & \vga\ GPUs & "
        r"GPUs saved & \$/month saved & "
        r"\fcfs\ util\% \\"
    )
    lines.append(r"\midrule")

    for lam in lambdas:
        r_fcfs = plan_capacity("opt-1.3b", "fcfs", lam)
        r_vga  = plan_capacity("opt-1.3b", "value_greedy_aging", lam)
        if not r_fcfs or not r_vga:
            continue

        saved = r_fcfs.min_gpus - r_vga.min_gpus
        saved_money = r_fcfs.cost_per_month - r_vga.cost_per_month

        fcfs_str = str(r_fcfs.min_gpus)
        vga_str  = str(r_vga.min_gpus)
        if not r_fcfs.sla_met:
            fcfs_str += r"$^\dagger$"
        if not r_vga.sla_met:
            vga_str += r"$^\dagger$"

        saved_str = (f"$-${saved}" if saved > 0
                     else "---")
        money_str = (f"\\${saved_money:,.0f}"
                     if saved_money > 0 else "---")
        util_str  = f"{r_fcfs.utilization_pct:.0f}\\%"

        # Bold the rows where savings occur
        if saved > 0:
            row = (f"\\textbf{{{lam}}} & "
                   f"\\textbf{{{fcfs_str}}} & "
                   f"\\textbf{{{vga_str}}} & "
                   f"\\textbf{{{saved_str}}} & "
                   f"\\textbf{{{money_str}}} & "
                   f"\\textbf{{{util_str}}} \\\\")
        else:
            row = (f"{lam} & {fcfs_str} & {vga_str} & "
                   f"{saved_str} & {money_str} & "
                   f"{util_str} \\\\")
        lines.append(row)

    lines.append(r"\midrule")

    # Summary row
    mu_fcfs = EMPIRICAL_MU[("opt-1.3b", "fcfs")]
    mu_vga  = EMPIRICAL_MU[("opt-1.3b", "value_greedy_aging")]
    lines.append(
        r"\multicolumn{6}{l}{\scriptsize "
        r"$^\dagger$SLA not met. "
        r"\fcfs\ $\mu_{\text{eff}}$ = "
        f"{mu_fcfs:.3f}\\,rps; "
        r"\vga\ $\mu_{\text{eff}}$ = "
        f"{mu_vga:.3f}\\,rps. "
        r"Savings window: "
        f"{mu_fcfs:.1f}--{mu_vga:.1f}\\,rps."
        r"}"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    run_analysis()
