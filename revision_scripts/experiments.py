#!/usr/bin/env python3
"""Embedded experiment definitions.

Each generator has the signature::

    fn(node_id: int, base_port: int, nodes: list[Node]) -> list[str]

It returns one command string per experiment configuration.  The runner
iterates through the list and executes each configuration in sequence.

Port assignment
---------------
Use ``base_port + i`` for the i-th configuration so that successive runs
never share a port even if the OS has not yet released the previous socket.
"""

from __future__ import annotations

import itertools
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List

# ---------------------------------------------------------------------------
# Path prefix (hard-coded; same inside Docker containers and on the host)
# ---------------------------------------------------------------------------

PREFIX = "/root/Lion-TKDE"
BIN_PREFIX = f"{PREFIX}/build"

# ---------------------------------------------------------------------------
# Node (kept here to avoid a circular import with nodes.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    internal_ip: str
    external_ip: str


@dataclass(frozen=True)
class AlgorithmConfig:
    protocol: str
    partitioner: str
    read_on_replica: bool = False
    random_router: bool = False
    migration_only: bool = False


# ---------------------------------------------------------------------------
# Default embedded IP list
# ---------------------------------------------------------------------------

EMBEDDED_IPS: list[Node] = [
    Node("10.10.10.73", "10.10.10.73"),
    Node("10.10.10.81", "10.10.10.81"),
    Node("10.10.10.82", "10.10.10.82"),
    Node("10.10.10.83", "10.10.10.83"),
    Node("10.10.10.2",  "10.10.10.2"),
]

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Mapping from human-readable algorithm name to AlgorithmConfig.
#
# Notes:
# - "Leap" is mapped to the LionS protocol family ("LIONS").
# - "Clay" / "LionS" / "LionSS" are MyClay variants with different partitioners.
# - "Lion" is a display-name alias for Clay (MyClay + Lion partitioner).
# - read_on_replica / random_router / migration_only only apply to Clay-family
#   protocols; hash-based protocols leave them at their False defaults.
ALGORITHMS: dict[str, AlgorithmConfig] = {
    "Clay"  : AlgorithmConfig("MyClay", "Lion", read_on_replica=0, random_router=1, migration_only=1),
    "Lion"  : AlgorithmConfig("Lion", "Lion", read_on_replica=1, random_router=0, migration_only=0),
    "Leap"  : AlgorithmConfig("Lion", "Lion", read_on_replica=0, random_router=1, migration_only=1),
    "Silo"  : AlgorithmConfig("SiloGC", "hash2", migration_only=1),
    "Aria"  : AlgorithmConfig("Aria", "hash"),
    "Calvin": AlgorithmConfig("Calvin", "hash"),
    "Star"  : AlgorithmConfig("Star", "hash"),
    "TwoPL" : AlgorithmConfig("TwoPLGC", "hash"),
}

CROSS_RATIOS = [0, 20, 50, 80, 100]

# Default parameters embedded for dist_ratio_ycsb_uniform
YCSB_UNIFORM_DEFAULT_ALGORITHMS: list[str] = ["Leap", "Silo", "Lion", "Clay"]
YCSB_UNIFORM_DEFAULT_CROSS_RATIOS: list[int] = [50]
DEFAULT_RUNTIME: int = 60

THREADS          = 12
PARTITION_PER_NODE = 12   # total partition_num = PARTITION_PER_NODE * len(nodes)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Partitioners that require Clay/Lion-specific flags
_CLAY_LION_PARTITIONERS = {"Lion", "LionS", "LionSS"}


def _skew_to_tag(skew_factor: float) -> int:
    """Convert skew representation into an integer tag for directory names.

    The codebase uses both [0, 1] (e.g. 0.9) and [0, 100] (e.g. 80) forms.
    """
    return int(round(skew_factor * 100)) if skew_factor <= 1.0 else int(round(skew_factor))


def _auto_data_src_path_dir(
    *,
    workload: str,
    skew_factor: float,
    dist_primary: int,
    dist_secondary: int | None = None,
) -> str:
    skew_tag = _skew_to_tag(skew_factor)
    if dist_secondary is None or dist_secondary == dist_primary:
        dist_tag = f"d{dist_primary}"
    else:
        dist_tag = f"d{dist_primary}_{dist_secondary}"
    return f'{PREFIX}/data/{workload}/s{skew_tag}_{dist_tag}_30/'


def _servers(node_id: int, port: int, nodes: list[Node]) -> str:
    """Build the semicolon-separated servers string for bench_ycsb / bench_tpcc.

    The current node uses its *internal* IP; all other nodes use their
    *external* IP.
    """
    parts = [
        f"{(node.internal_ip if idx == node_id else node.external_ip)}:{port}"
        for idx, node in enumerate(nodes)
    ]
    return ";".join(parts)


def _ycsb_cmd(
    node_id: int,
    servers: str,
    n_nodes: int,
    cfg: AlgorithmConfig,
    *,
    cross_ratio: int = 0,
    skew_factor: float = 0.0,
    data_src_path_dir: str | None = None,
) -> str:
    partition_num = PARTITION_PER_NODE * n_nodes
    skew_value = _skew_to_tag(skew_factor)
    cmd = (
        f"{BIN_PREFIX}/bench_ycsb --logtostderr=1 --id={node_id} "
        f'--servers="{servers}" '
        f"--protocol={cfg.protocol} "
        f"--partition_num={partition_num} "
        f"--threads={THREADS} "
        f"--partitioner={cfg.partitioner} "
        f"--read_write_ratio=90 "
        f"--cross_ratio={cross_ratio} "
        f"--skew_factor={skew_value}"
    )
    if cfg.partitioner in _CLAY_LION_PARTITIONERS:
        data_dir = data_src_path_dir or _auto_data_src_path_dir(
            workload="ycsb",
            skew_factor=skew_factor,
            dist_primary=cross_ratio,
        )
        cmd += (
            " --batch_size=10000"
            " --batch_flush=500"
            " --lion_with_metis_init=1"
            " --time_to_run=60"
            " --workload_time=60"
            " --sample_time_interval=3"
            f" --migration_only={int(cfg.migration_only)}"
            f" --read_on_replica={'true' if cfg.read_on_replica else 'false'}"
            f" --random_router={int(cfg.random_router)}"
            f' --data_src_path_dir="{data_dir}"'
        )
    return cmd


def _tpcc_cmd(
    node_id: int,
    servers: str,
    n_nodes: int,
    cfg: AlgorithmConfig,
    *,
    neworder_dist: int = 0,
    payment_dist: int = 0,
    skew_factor: float = 0.0,
    data_src_path_dir: str | None = None,
) -> str:
    partition_num = PARTITION_PER_NODE * n_nodes
    skew_value = _skew_to_tag(skew_factor)
    cmd = (
        f"{BIN_PREFIX}/bench_tpcc --logtostderr=1 --id={node_id} "
        f'--servers="{servers}" '
        f"--protocol={cfg.protocol} "
        f"--partition_num={partition_num} "
        f"--threads={THREADS} "
        f"--partitioner={cfg.partitioner} "
        f"--query=mixed "
        f"--neworder_dist={neworder_dist} "
        f"--payment_dist={payment_dist} "
        f"--skew_factor={skew_value}"
    )
    if cfg.partitioner in _CLAY_LION_PARTITIONERS:
        data_dir = data_src_path_dir or _auto_data_src_path_dir(
            workload="tpcc",
            skew_factor=skew_factor,
            dist_primary=neworder_dist,
            dist_secondary=payment_dist,
        )
        cmd += (
            " --batch_size=10000"
            " --batch_flush=500"
            " --lion_with_metis_init=1"
            " --time_to_run=60"
            " --workload_time=60"
            " --sample_time_interval=3"
            f" --migration_only={int(cfg.migration_only)}"
            f" --read_on_replica={'true' if cfg.read_on_replica else 'false'}"
            f" --random_router={int(cfg.random_router)}"
            f' --data_src_path_dir="{data_dir}"'
        )
    return cmd


def _pps_cmd(
    node_id: int,
    servers: str,
    n_nodes: int,
    cfg: AlgorithmConfig,
    *,
    cross_ratio: int = 0,
    zipf: float = 0.0,
    read_write_ratio: int = 80,
) -> str:
    partition_num = PARTITION_PER_NODE * n_nodes
    cmd = (
        f"{BIN_PREFIX}/bench_pps --logtostderr=1 --id={node_id} "
        f'--servers="{servers}" '
        f"--protocol={cfg.protocol} "
        f"--partition_num={partition_num} "
        f"--threads={THREADS} "
        f"--partitioner={cfg.partitioner} "
        f"--cross_ratio={cross_ratio} "
        f"--read_write_ratio={read_write_ratio} "
        f"--zipf={zipf}"
    )
    if cfg.partitioner in _CLAY_LION_PARTITIONERS:
        cmd += (
            " --batch_size=10000"
            " --batch_flush=500"
            " --lion_with_metis_init=1"
            " --time_to_run=60"
            " --workload_time=60"
            " --sample_time_interval=3"
            f" --migration_only={int(cfg.migration_only)}"
            f" --read_on_replica={'true' if cfg.read_on_replica else 'false'}"
            f" --random_router={int(cfg.random_router)}"
        )
    return cmd


def _resolve_algorithms(algorithms: list[str] | None) -> list[str]:
    if algorithms is None:
        return list(ALGORITHMS.keys())
    unknown = [name for name in algorithms if name not in ALGORITHMS]
    if unknown:
        valid = ", ".join(sorted(ALGORITHMS))
        bad = ", ".join(unknown)
        raise ValueError(f"Unknown algorithm(s): {bad}. Valid values: {valid}")
    return algorithms


# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------

GeneratorFn = Callable[[int, int, List[Node]], List[str]]


def dist_ratio_ycsb_uniform(
    node_id: int,
    base_port: int,
    nodes: list[Node],
    *,
    algorithms: list[str] | None = None,
    cross_ratios: list[int] | None = None,
    skew_factor: float = 0.0,
) -> list[str]:
    """YCSB uniform: cross-partition ratio sweep × algorithm.

    Args:
        algorithms: Algorithm list. Defaults to YCSB_UNIFORM_DEFAULT_ALGORITHMS.
        cross_ratios: Distributed ratio list. Defaults to YCSB_UNIFORM_DEFAULT_CROSS_RATIOS.
        skew_factor: Keep 0.0 for uniform, override only if needed.
    """
    algorithms_ = _resolve_algorithms(
        algorithms if algorithms is not None else YCSB_UNIFORM_DEFAULT_ALGORITHMS
    )
    ratios = (
        cross_ratios if cross_ratios is not None else YCSB_UNIFORM_DEFAULT_CROSS_RATIOS
    )
    cmds = []
    for i, (algo, ratio) in enumerate(itertools.product(algorithms_, ratios)):
        cfg = ALGORITHMS[algo]
        servers = _servers(node_id, base_port + i, nodes)
        cmds.append(
            _ycsb_cmd(
                node_id, servers, len(nodes), cfg,
                cross_ratio=ratio, skew_factor=skew_factor,
            )
        )
    return cmds


def dist_ratio_ycsb_skew(
    node_id: int,
    base_port: int,
    nodes: list[Node],
    *,
    algorithms: list[str] | None = None,
    cross_ratios: list[int] | None = None,
    skew_factor: float = 0.9,
) -> list[str]:
    """YCSB skewed (Zipfian): cross-partition ratio sweep × algorithm."""
    algorithms_ = _resolve_algorithms(algorithms)
    ratios = CROSS_RATIOS if cross_ratios is None else cross_ratios
    cmds = []
    for i, (algo, ratio) in enumerate(itertools.product(algorithms_, ratios)):
        cfg = ALGORITHMS[algo]
        servers = _servers(node_id, base_port + i, nodes)
        cmds.append(
            _ycsb_cmd(
                node_id, servers, len(nodes), cfg,
                cross_ratio=ratio, skew_factor=skew_factor,
            )
        )
    return cmds


def dist_ratio_tpcc(
    node_id: int,
    base_port: int,
    nodes: list[Node],
    *,
    algorithms: list[str] | None = None,
    cross_ratios: list[int] | None = None,
    distributions: list[str] | None = None,
    skew_for_skew_dist: float = 0.9,
    skew_for_normal_dist: float = 0.0,
) -> list[str]:
    """TPC-C: distribution-ratio sweep × distribution type × algorithm.

    ``distribution`` controls whether warehouse accesses are skewed (some
    warehouses are much hotter) or uniformly distributed.
    """
    distribution = ["skew", "normal"] if distributions is None else distributions
    unknown_dist = [dist for dist in distribution if dist not in {"skew", "normal"}]
    if unknown_dist:
        bad = ", ".join(unknown_dist)
        raise ValueError(f"Unknown distribution type(s): {bad}. Valid values: skew, normal")

    algorithms_ = _resolve_algorithms(algorithms)
    ratios = CROSS_RATIOS if cross_ratios is None else cross_ratios
    cmds = []
    for i, (dist, algo, ratio) in enumerate(
        itertools.product(distribution, algorithms_, ratios)
    ):
        cfg = ALGORITHMS[algo]
        servers = _servers(node_id, base_port + i, nodes)
        skew_factor = skew_for_skew_dist if dist == "skew" else skew_for_normal_dist
        cmds.append(
            _tpcc_cmd(
                node_id, servers, len(nodes), cfg,
                neworder_dist=ratio, payment_dist=ratio,
                skew_factor=skew_factor,
            )
        )
    return cmds


def dist_ratio_pps(
    node_id: int,
    base_port: int,
    nodes: list[Node],
    *,
    algorithms: list[str] | None = None,
    cross_ratios: list[int] | None = None,
    distributions: list[str] | None = None,
    zipf_for_skew_dist: float = 0.9,
    zipf_for_normal_dist: float = 0.0,
    read_write_ratio: int = 80,
) -> list[str]:
    """PPS: distribution-ratio sweep × distribution type × algorithm."""
    dist_list = ["skew", "normal"] if distributions is None else distributions
    unknown_dist = [dist for dist in dist_list if dist not in {"skew", "normal"}]
    if unknown_dist:
        bad = ", ".join(unknown_dist)
        raise ValueError(f"Unknown distribution type(s): {bad}. Valid values: skew, normal")

    algorithms_ = _resolve_algorithms(algorithms)
    ratios = CROSS_RATIOS if cross_ratios is None else cross_ratios
    cmds = []
    for i, (dist, algo, ratio) in enumerate(itertools.product(dist_list, algorithms_, ratios)):
        cfg = ALGORITHMS[algo]
        servers = _servers(node_id, base_port + i, nodes)
        zipf = zipf_for_skew_dist if dist == "skew" else zipf_for_normal_dist
        cmds.append(
            _pps_cmd(
                node_id,
                servers,
                len(nodes),
                cfg,
                cross_ratio=ratio,
                zipf=zipf,
                read_write_ratio=read_write_ratio,
            )
        )
    return cmds


# ---------------------------------------------------------------------------
# Experiment label helpers (mirror iteration order of each generator)
# ---------------------------------------------------------------------------


def get_experiment_labels(
    generator_name: str,
    *,
    algorithms: list[str] | None = None,
    cross_ratios: list[int] | None = None,
    distributions: list[str] | None = None,
) -> list[str]:
    """Return one short label per experiment, in the same order as the commands.

    Labels are used to construct meaningful result directory names, e.g.
    ``exp_0003_Leap_cross80``.
    """
    if generator_name == "dist_ratio_ycsb_uniform":
        algos = _resolve_algorithms(
            algorithms if algorithms is not None else YCSB_UNIFORM_DEFAULT_ALGORITHMS
        )
        ratios = cross_ratios if cross_ratios is not None else YCSB_UNIFORM_DEFAULT_CROSS_RATIOS
        return [f"{algo}_cross{ratio}" for algo, ratio in itertools.product(algos, ratios)]

    if generator_name == "dist_ratio_ycsb_skew":
        algos = _resolve_algorithms(algorithms)
        ratios = CROSS_RATIOS if cross_ratios is None else cross_ratios
        return [f"{algo}_cross{ratio}" for algo, ratio in itertools.product(algos, ratios)]

    if generator_name == "dist_ratio_tpcc":
        dist_list = ["skew", "normal"] if distributions is None else distributions
        algos = _resolve_algorithms(algorithms)
        ratios = CROSS_RATIOS if cross_ratios is None else cross_ratios
        return [
            f"{dist}_{algo}_cross{ratio}"
            for dist, algo, ratio in itertools.product(dist_list, algos, ratios)
        ]

    if generator_name == "dist_ratio_pps":
        dist_list = ["skew", "normal"] if distributions is None else distributions
        algos = _resolve_algorithms(algorithms)
        ratios = CROSS_RATIOS if cross_ratios is None else cross_ratios
        return [
            f"{dist}_{algo}_cross{ratio}"
            for dist, algo, ratio in itertools.product(dist_list, algos, ratios)
        ]

    # Fallback: numeric labels (caller must supply count via algorithms/ratios)
    return []


# ---------------------------------------------------------------------------
# Registry — add new generators here
# ---------------------------------------------------------------------------

EMBEDDED_GENERATORS: dict[str, GeneratorFn] = {
    "dist_ratio_ycsb_uniform": dist_ratio_ycsb_uniform,
    "dist_ratio_ycsb_skew"   : dist_ratio_ycsb_skew,
    "dist_ratio_tpcc"        : dist_ratio_tpcc,
    "dist_ratio_pps"         : dist_ratio_pps,
}


# ---------------------------------------------------------------------------
# Command-file utilities
# ---------------------------------------------------------------------------

def generate_commands(
    generator_name: str,
    node_id: int,
    base_port: int,
    nodes: list[Node],
    params: dict[str, Any] | None = None,
) -> list[str]:
    """Return commands for *node_id* across all experiment configs.

    The optional ``params`` dictionary lets callers explicitly control
    per-generator parameters such as algorithm list, distributed ratios,
    and distribution behavior.
    """
    fn = EMBEDDED_GENERATORS[generator_name]
    if not params:
        return fn(node_id, base_port, nodes)
    allowed = set(inspect.signature(fn).parameters.keys())
    filtered_params = {
        key: value for key, value in params.items() if key in allowed
    }
    return fn(node_id, base_port, nodes, **filtered_params)


def save_node_commands(commands: list[str], path: Path, dry_run: bool) -> None:
    """Write all commands for a node to a single file, one per line."""
    print(f"Saving {len(commands)} command(s) to {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")
