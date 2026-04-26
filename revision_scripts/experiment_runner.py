#!/usr/bin/env python3
"""Distribute experiment commands, execute them, and collect results.

Module layout
-------------
experiments.py           -- PREFIX, ALGORITHMS, Node, IPs, generator functions
nodes.py                 -- read_nodes (file or embedded)
generators.py            -- render_generated_commands, save_node_commands
ssh_utils.py             -- run_cmd, ssh_run, scp_from_remote, container helpers
execution.py             -- prepare / start / stop nodes (remote and local)
results.py               -- prepare_result_dirs, collect_results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from experiments import EMBEDDED_GENERATORS, PREFIX, generate_commands, save_node_commands
from execution import (
    prepare_remote_nodes,
    start_local,
    start_remote_nodes,
    stop_local,
    stop_remote_nodes,
)
from nodes import read_nodes
from results import collect_results, prepare_result_dirs


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_REMOTE_DATA_DIR   = f"{PREFIX}/data"
DEFAULT_LOCAL_COMMIT_DIR  = f"{PREFIX}/data/commit"
DEFAULT_LOCAL_RESULTS_ROOT = f"{PREFIX}/data/results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run distributed experiments and collect results."
    )

    # --- generator -----------------------------------------------------------
    parser.add_argument(
        "--generator",
        choices=sorted(EMBEDDED_GENERATORS),
        required=True,
        help="Name of the experiment generator in experiments.py.",
    )
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated algorithms, e.g. 'Leap,Silo,Clay,TwoPL'.",
    )
    parser.add_argument(
        "--cross-ratios",
        default=None,
        help="Comma-separated ratios, e.g. '0,20,50,80,100'.",
    )
    parser.add_argument(
        "--distributions",
        default=None,
        help="Comma-separated distributions for generators that support them, e.g. 'normal' or 'skew,normal'.",
    )
    parser.add_argument(
        "--skew-factor",
        type=float,
        default=None,
        help="Override skew_factor for YCSB generators only.",
    )

    # --- cluster topology ----------------------------------------------------
    parser.add_argument(
        "--local-index",
        type=int,
        default=None,
        help="Index of the local node. Defaults to the last entry.",
    )

    # --- connection ----------------------------------------------------------
    parser.add_argument("--port",        type=int, default=22010,
                        help="Base port passed to the generator (incremented per config).")
    parser.add_argument("--remote-user", default="zhanhao",
                        help="SSH user for remote machines.")
    parser.add_argument("--ssh-key",     default=None,
                        help="SSH private key path. Defaults to null (use SSH default config/agent).")

    # --- Docker container names ----------------------------------------------
    parser.add_argument("--local-container", default="lion",
                        help="Local Docker container name used to run local node commands.")
    parser.add_argument("--container-prefix", default="lion_",
                        help="Remote container prefix; node i uses <prefix><i+1>.")
    parser.add_argument("--container-names", nargs="*", default=None,
                        help="Explicit container names (overrides --container-prefix).")

    # --- result collection ---------------------------------------------------
    parser.add_argument("--remote-data-dir",   default=DEFAULT_REMOTE_DATA_DIR,
                        help="Remote directory to collect result files from.")
    parser.add_argument("--local-commit-dir",  default=DEFAULT_LOCAL_COMMIT_DIR,
                        help="Local staging directory for collected result files.")
    parser.add_argument("--results-root",      default=DEFAULT_LOCAL_RESULTS_ROOT,
                        help="Root directory for final experiment outputs.")
    parser.add_argument("--result-pattern",    default="commits_{id}.xls",
                        help="Remote result filename pattern (uses {id} placeholder).")

    # --- misc ----------------------------------------------------------------
    parser.add_argument("--runtime", type=int, required=True,
                        help="Seconds to run each experiment configuration.")
    parser.add_argument("--bin-name", default="bench_ycsb",
                        help="Process name to kill before and after each run.")
    parser.add_argument("--save-generated-dir",
                        default=f"{PREFIX}/revision_scripts/generated_commands",
                        help="Directory to save the rendered per-node command files.")
    parser.add_argument("--skip-remote-clean", action="store_true",
                        help="Skip pre/post cleanup on remote hosts.")
    parser.add_argument("--skip-collect",      action="store_true",
                        help="Skip pulling result files after each run.")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Print commands without executing them.")

    return parser.parse_args()


def _parse_csv_str(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [token.strip() for token in value.split(",") if token.strip()]
    return out or None


def _parse_csv_int(value: str | None) -> list[int] | None:
    tokens = _parse_csv_str(value)
    if tokens is None:
        return None
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid integer list: {value}") from exc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Accept common null-like values from CLI wrappers.
    if args.ssh_key in {"", "null", "None"}:
        args.ssh_key = None

    save_generated_dir = Path(args.save_generated_dir).expanduser()
    local_commit_dir   = Path(args.local_commit_dir).expanduser()
    results_root       = Path(args.results_root).expanduser()

    generator_params: dict[str, Any] = {}
    algorithms = _parse_csv_str(args.algorithms)
    cross_ratios = _parse_csv_int(args.cross_ratios)
    distributions = _parse_csv_str(args.distributions)
    if algorithms is not None:
        generator_params["algorithms"] = algorithms
    if cross_ratios is not None:
        generator_params["cross_ratios"] = cross_ratios
    if distributions is not None:
        generator_params["distributions"] = distributions
    if args.skew_factor is not None:
        generator_params["skew_factor"] = args.skew_factor

    # --- load nodes ----------------------------------------------------------
    nodes = read_nodes()
    local_index = args.local_index if args.local_index is not None else len(nodes) - 1
    if not (0 <= local_index < len(nodes)):
        raise ValueError(
            f"--local-index {local_index} is out of range for {len(nodes)} nodes"
        )

    remote_nodes = [
        (idx, node) for idx, node in enumerate(nodes) if idx != local_index
    ]

    if args.container_names and len(args.container_names) != len(remote_nodes):
        raise ValueError(
            f"--container-names must have {len(remote_nodes)} entries "
            f"(one per remote node), got {len(args.container_names)}"
        )

    print(f"Loaded {len(nodes)} nodes  (local index: {local_index} — "
          f"{nodes[local_index].external_ip})")

    # --- generate all commands -----------------------------------------------
    all_remote_commands = prepare_remote_nodes(
        remote_nodes,
        generator_name=args.generator,
        base_port=args.port,
        nodes=nodes,
        save_generated_dir=save_generated_dir,
        remote_user=args.remote_user,
        ssh_key=args.ssh_key,
        container_prefix=args.container_prefix,
        container_names=args.container_names,
        bin_name=args.bin_name,
        skip_remote_clean=args.skip_remote_clean,
        dry_run=args.dry_run,
        generator_params=generator_params or None,
    )

    local_commands = generate_commands(
        args.generator,
        local_index,
        args.port,
        nodes,
        params=generator_params or None,
    )
    save_node_commands(
        local_commands, save_generated_dir / f"local_{local_index}.sh", args.dry_run
    )

    num_experiments = len(local_commands)
    print(f"Generator '{args.generator}' produced {num_experiments} configuration(s).")

    # --- run each configuration in sequence ----------------------------------
    result_dir = prepare_result_dirs(
        args.generator, results_root, local_commit_dir, args.dry_run
    )

    for exp_idx in range(num_experiments):
        print(f"\n=== Experiment {exp_idx + 1}/{num_experiments} ===")

        exp_remote_commands = {
            node_id: cmds[exp_idx]
            for node_id, cmds in all_remote_commands.items()
        }
        exp_local_command = local_commands[exp_idx]

        try:
            start_remote_nodes(
                remote_nodes,
                commands=exp_remote_commands,
                remote_user=args.remote_user,
                ssh_key=args.ssh_key,
                container_prefix=args.container_prefix,
                container_names=args.container_names,
                dry_run=args.dry_run,
            )
            start_local(exp_local_command, args.local_container, args.dry_run)
            print(f"Waiting {args.runtime}s …")
            if not args.dry_run:
                time.sleep(args.runtime)
        finally:
            stop_local(args.bin_name, args.local_container, args.dry_run)
            stop_remote_nodes(
                remote_nodes,
                remote_user=args.remote_user,
                ssh_key=args.ssh_key,
                container_prefix=args.container_prefix,
                container_names=args.container_names,
                bin_name=args.bin_name,
                skip_remote_clean=args.skip_remote_clean,
                dry_run=args.dry_run,
            )

        if not args.skip_collect:
            exp_result_dir  = result_dir / f"exp_{exp_idx:04d}"
            exp_commit_dir  = local_commit_dir / f"exp_{exp_idx:04d}"
            collect_results(
                remote_nodes,
                remote_user=args.remote_user,
                ssh_key=args.ssh_key,
                remote_data_dir=args.remote_data_dir,
                result_pattern=args.result_pattern,
                local_commit_dir=exp_commit_dir,
                result_dir=exp_result_dir,
                dry_run=args.dry_run,
            )

    print(f"\nAll {num_experiments} experiment(s) completed.")
    print(f"Results: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
