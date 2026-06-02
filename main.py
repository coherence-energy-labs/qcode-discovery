"""CLI entry point for running the BB code evaluation pipeline.

Generates candidate polynomial pairs using the seed
``generate_candidates`` function (or whatever is currently in
``evolve/seed_solution.py``), evaluates them through the multi-stage
cascade (see :mod:`evaluation.evaluator`), and saves the best results
to ``results/discovered_codes.json`` and ``results/pareto_front.json``.

Each ``(ell, m)`` lattice is evaluated as a separate "generation" for
tracking purposes, with per-lattice summaries logged to
``results/runs/<run_id>/generations.jsonl``.

Usage::

    # Quick k-only scan across all 18 target lattices (~30 s)
    uv run python main.py --quick

    # Evaluate specific lattices with distance estimation
    uv run python main.py --lattices 12,6 6,6

    # Full cascade with custom thresholds
    uv run python main.py --lattices 12,6 --quick-trials 200 --fom-threshold-refine 5.0

    # Verbose output
    uv run python main.py --lattices 6,6 --quick -v

CLI options::

    --lattices ELL,M ...      Lattice dimensions (default: all 18 target lattices)
    --quick                   Compute k only, skip distance estimation
    --quick-trials N          BP-OSD trials for initial distance estimate (default: 100)
    --refine-trials N         BP-OSD trials for refined estimate (default: 1000)
    --fom-threshold-refine F  FOM threshold for refined estimation (default: 6.0)
    --fom-threshold-exact F   FOM threshold for exact distance (default: 8.0)
    --top N                   Number of top results to display (default: 10)
    --run-id ID               Run identifier for tracking (auto-generated if not set)
    -v, --verbose             Enable debug logging
"""

from __future__ import annotations

import argparse
import logging

from evaluation.evaluator import evaluate_batch
from evaluation.results import save_code, update_pareto_front
from evaluation.tracking import RunTracker
from evolve.seed_solution import TARGET_LATTICES, generate_candidates

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bivariate bicycle codes across target lattices."
    )
    parser.add_argument(
        "--lattices", nargs="*", type=str, default=None,
        help="Lattice dimensions as 'ell,m' pairs (e.g. 12,6 9,8). "
             "Defaults to all target lattices.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: compute k only, skip distance estimation.",
    )
    parser.add_argument(
        "--quick-trials", type=int, default=100,
        help="Number of BP-OSD trials for initial distance estimate.",
    )
    parser.add_argument(
        "--refine-trials", type=int, default=1000,
        help="Number of BP-OSD trials for refined distance estimate.",
    )
    parser.add_argument(
        "--fom-threshold-refine", type=float, default=6.0,
        help="FOM threshold to trigger refined distance estimation.",
    )
    parser.add_argument(
        "--fom-threshold-exact", type=float, default=8.0,
        help="FOM threshold to trigger exact distance computation.",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top results to display.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier for tracking. Auto-generated if not set.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.lattices:
        lattices = []
        for spec in args.lattices:
            ell, m = spec.split(",")
            lattices.append((int(ell), int(m)))
    else:
        lattices = TARGET_LATTICES

    eval_kwargs = dict(
        quick=args.quick,
        quick_trials=args.quick_trials,
        refine_trials=args.refine_trials,
        fom_threshold_refine=args.fom_threshold_refine,
        fom_threshold_exact=args.fom_threshold_exact,
    )

    # Initialize run tracker
    tracker = RunTracker()
    run_id = tracker.start_run(
        run_id=args.run_id,
        config={"lattices": lattices, **eval_kwargs},
    )
    print(f"Run {run_id}: evaluating {len(lattices)} lattice(s)...")

    # Evaluate each lattice as a "generation" for tracking
    all_results = []
    for gen_idx, (ell, m) in enumerate(lattices):
        candidates = generate_candidates(ell, m)
        logger.info(
            "Lattice (%d, %d): %d candidates, n=%d",
            ell, m, len(candidates), 2 * ell * m,
        )

        tracker.start_generation(gen_idx)
        results = evaluate_batch(ell, m, candidates, **eval_kwargs)
        for r in results:
            tracker.log_evaluation(r)
        summary = tracker.end_generation(gen_idx, results)

        all_results.extend(results)
        print(
            f"  ({ell},{m}) n={2*ell*m}: "
            f"{summary['valid_candidates']}/{summary['total_candidates']} valid, "
            f"best FOM={summary['best_fom']:.2f}"
        )

    run_meta = tracker.end_run()

    # Save and display top results
    top = [r for r in all_results if r.get("score", 0) > 0]
    top.sort(key=lambda r: r["score"], reverse=True)
    top = top[:args.top]
    for r in top:
        save_code(r)
    if top:
        update_pareto_front(top)

    print(f"\nTop {min(len(top), args.top)} codes found:")
    print(f"{'Code':>20s}  {'FOM':>6s}  {'Stage':>16s}")
    print("-" * 48)
    for r in top:
        label = f"[[{r['n']},{r['k']},{r['d']}]]"
        print(f"{label:>20s}  {r['fom']:6.2f}  {r['stage']:>16s}")

    if not top:
        print("  (no codes with positive score found)")

    print(f"\nRun log: {tracker.run_dir}/")
    print(f"  Total evaluations: {run_meta['total_evaluations']}")
    print(f"  Best FOM: {run_meta['best_fom']:.2f}")


if __name__ == "__main__":
    main()
