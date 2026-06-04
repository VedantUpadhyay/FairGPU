import random
from dataclasses import dataclass

import numpy as np


@dataclass
class TraceRequest:
    request_id: str
    prompt: str
    sla_tier: str
    arrival_time: float
    priority: int


def load_mixed_trace(
    n_interactive: int = 300, n_batch: int = 700, seed: int = 42
) -> list[TraceRequest]:
    """
    Returns mixed trace sorted by arrival_time.

    Interactive: from ShareGPT (short chat prompts)
    Batch: from GovReport (long document summaries)
    Falls back to synthetic data if download fails.

    Arrival times: Poisson process, lambda=10 req/sec
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # --- Load interactive prompts ---
    interactive_prompts = []
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            split="train",
            streaming=True,
        )
        for item in ds:
            for turn in item.get("conversations", []):
                if turn.get("from") == "human":
                    txt = turn.get("value", "")
                    # Keep short prompts only
                    if 10 < len(txt.split()) < 200:
                        interactive_prompts.append(txt)
            if len(interactive_prompts) >= n_interactive * 3:
                break
    except Exception as e:
        print(f"ShareGPT load failed ({e}), " f"using synthetic interactive prompts")

    if len(interactive_prompts) < n_interactive:
        # Synthetic fallback
        templates = [
            "What is {}?",
            "Explain {} in simple terms.",
            "How do I {}?",
            "What are the benefits of {}?",
            "Can you summarize {}?",
        ]
        topics = [
            "machine learning",
            "Python",
            "cloud computing",
            "scheduling",
            "neural networks",
            "databases",
            "operating systems",
            "networking",
        ]
        while len(interactive_prompts) < n_interactive * 3:
            t = random.choice(templates)
            o = random.choice(topics)
            interactive_prompts.append(t.format(o))

    # --- Load batch prompts ---
    batch_prompts = []
    try:
        from datasets import load_dataset

        ds = load_dataset("ccdv/govreport-summarization", split="train", streaming=True)
        for item in ds:
            txt = item.get("report", "")
            words = txt.split()
            if len(words) > 200:
                # Truncate to 800 words
                batch_prompts.append(" ".join(words[:800]))
            if len(batch_prompts) >= n_batch * 3:
                break
    except Exception as e:
        print(f"GovReport load failed ({e}), " f"using synthetic batch prompts")

    if len(batch_prompts) < n_batch:
        # Synthetic fallback - long paragraphs
        sentences = [
            "The government report details the "
            "findings of the committee on "
            "infrastructure spending. ",
            "Analysis of the data reveals "
            "significant trends in public policy. ",
            "The committee recommends further "
            "investigation into these matters. ",
        ]
        while len(batch_prompts) < n_batch * 3:
            # Build ~600 word prompt
            para = " ".join(random.choices(sentences, k=60))
            batch_prompts.append(para)

    # --- Sample ---
    interactive_sample = random.sample(
        interactive_prompts, min(n_interactive, len(interactive_prompts))
    )
    batch_sample = random.sample(batch_prompts, min(n_batch, len(batch_prompts)))

    # --- Assign Poisson arrival times ---
    n_total = len(interactive_sample) + len(batch_sample)
    inter_arrivals = rng.exponential(1 / 10, n_total)
    arrival_times = np.cumsum(inter_arrivals).tolist()

    # --- Build requests ---
    requests = []
    all_prompts = (
        [(p, "interactive", 1) for p in interactive_sample]
        + [(p, "batch", 0) for p in batch_sample]
    )
    # Shuffle then assign arrival times
    shuffled_idx = list(range(len(all_prompts)))
    random.shuffle(shuffled_idx)

    for rank, idx in enumerate(shuffled_idx):
        prompt, tier, priority = all_prompts[idx]
        requests.append(
            TraceRequest(
                request_id=f"{tier}-{idx:04d}",
                prompt=prompt,
                sla_tier=tier,
                arrival_time=arrival_times[rank],
                priority=priority,
            )
        )

    requests.sort(key=lambda r: r.arrival_time)
    print(
        f"Trace loaded: {len(interactive_sample)} "
        f"interactive + {len(batch_sample)} batch "
        f"= {len(requests)} total requests"
    )
    return requests
