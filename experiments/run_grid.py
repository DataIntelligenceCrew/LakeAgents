#!/usr/bin/env python3
"""
Grid search over (τ, β) for perturbation benchmark.
τ (threshold) ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
β (beta)      ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
"""
import sys
import argparse
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from benchmark_perturbation.benchmark_perturbation import run_full_pipeline

TAU_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]


def run_grid(verbose: bool = True):
    """Run perturbation pipeline for all (τ, β) combinations."""
    results = []
    total = len(TAU_VALUES) * len(BETA_VALUES)
    idx = 0
    for tau in TAU_VALUES:
        for beta in BETA_VALUES:
            idx += 1
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{idx}/{total}] τ={tau}, β={beta}")
                print("="*60)
            try:
                run_full_pipeline(threshold=tau, beta=beta)
                results.append({"tau": tau, "beta": beta, "status": "ok"})
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"tau": tau, "beta": beta, "status": "error", "error": str(e)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Grid search over (τ, β) for perturbation")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    print(f"Grid: τ ∈ {TAU_VALUES}, β ∈ {BETA_VALUES}")
    print(f"Total: {len(TAU_VALUES) * len(BETA_VALUES)} combinations")
    print(f"Output dirs: perturbed_{{τ}}_{{β}}")

    results = run_grid(verbose=not args.quiet)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone: {ok_count}/{len(results)} succeeded")
    for r in results:
        if r["status"] != "ok":
            print(f"  Failed τ={r['tau']}, β={r['beta']}: {r.get('error', '')}")


if __name__ == "__main__":
    main()
