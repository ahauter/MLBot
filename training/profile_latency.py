#!/usr/bin/env python3
"""
Inference Latency Profiler
==========================
Measures encoder + policy forward pass latency to verify the 8ms budget
at 120Hz is met with T_WINDOW=8 frame history.

Usage
-----
    python training/profile_latency.py
    python training/profile_latency.py --t-window 4   # compare T=4 vs T=8
    python training/profile_latency.py --device cuda   # profile on GPU
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))

from encoder import (
    SharedTransformerEncoder,
    ENTITY_TYPE_IDS_1V1,
    N_TOKENS,
    TOKEN_FEATURES,
    D_MODEL,
)
from policy_head import PolicyHead


def profile(t_window: int, device_str: str, n_warmup: int = 200, n_trials: int = 1000):
    device = torch.device(device_str)
    encoder = SharedTransformerEncoder().to(device).eval()
    policy = PolicyHead().to(device).eval()
    entity_ids = torch.tensor(ENTITY_TYPE_IDS_1V1, dtype=torch.long, device=device)

    dummy = torch.randn(1, t_window, N_TOKENS, TOKEN_FEATURES, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            emb = encoder(dummy, entity_ids)
            policy(emb)

    # Timed runs
    if device_str == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_trials):
            start = time.perf_counter()
            emb = encoder(dummy, entity_ids)
            policy(emb)
            if device_str == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    times_ms = np.array(times) * 1000.0
    print(f'\nInference Latency Profile (T_WINDOW={t_window}, device={device_str})')
    print(f'  Trials:  {n_trials}')
    print(f'  Mean:    {times_ms.mean():.3f} ms')
    print(f'  Median:  {np.median(times_ms):.3f} ms')
    print(f'  P95:     {np.percentile(times_ms, 95):.3f} ms')
    print(f'  P99:     {np.percentile(times_ms, 99):.3f} ms')
    print(f'  Min:     {times_ms.min():.3f} ms')
    print(f'  Max:     {times_ms.max():.3f} ms')

    budget = 8.0
    if times_ms.mean() < budget:
        print(f'\n  PASS: Mean {times_ms.mean():.3f}ms < {budget}ms budget')
    else:
        print(f'\n  FAIL: Mean {times_ms.mean():.3f}ms >= {budget}ms budget')

    return times_ms


def main():
    parser = argparse.ArgumentParser(description='Profile inference latency.')
    parser.add_argument('--t-window', type=int, default=8)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--trials', type=int, default=1000)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU.')
        args.device = 'cpu'

    profile(args.t_window, args.device, n_trials=args.trials)


if __name__ == '__main__':
    main()
