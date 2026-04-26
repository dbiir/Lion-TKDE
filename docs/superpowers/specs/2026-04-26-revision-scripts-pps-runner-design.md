# Revision Scripts PPS Runner Design

## Summary

Add PPS workload support to the `revision_scripts` experiment system by introducing a new embedded generator in `revision_scripts/experiments.py`. The new generator will produce PPS experiment commands using the same orchestration path already used for YCSB and TPC-C, without adding any new top-level runner script.

This design treats `revision_scripts` as the source of truth and does not extend the legacy `scripts/` command generators.

## Goals

- Add a PPS workload runner to the `revision_scripts` system.
- Preserve the existing generic orchestration flow in `revision_scripts/experiment_runner.py`.
- Support experiment sweeps over `algorithm × cross_ratio × distribution`.
- Map distributions to PPS Zipf behavior using defaults that match the desired runtime behavior.
- Keep the new workload-specific logic isolated and easy to understand.

## Non-Goals

- Do not modify or revive the legacy `scripts/.../aws_*.py` structure.
- Do not refactor the generic runner architecture.
- Do not introduce a new CLI entrypoint for PPS.
- Do not change existing TPC-C or YCSB generator behavior beyond any strictly necessary shared helper reuse.

## Existing Context

The active runner system lives in `revision_scripts/`:

- `experiment_runner.py` parses CLI arguments, generates commands for each node, starts experiments, waits for runtime, stops processes, and collects results.
- `experiments.py` defines the embedded experiment generators and command builders.
- Existing generators include `dist_ratio_ycsb_uniform`, `dist_ratio_ycsb_skew`, and `dist_ratio_tpcc`.

The PPS benchmark already exists in the codebase through `bench_pps.cpp` and `benchmark/pps/*`, so the missing piece is the command-generation layer inside `revision_scripts`.

PPS exposes the following relevant workload flags:

- `--cross_ratio`
- `--zipf`
- `--read_write_ratio`
- `--keys`

For this design, the key workload knobs are `cross_ratio` and `zipf`.

## Proposed Approach

Add PPS as a first-class generator in `revision_scripts/experiments.py` by introducing:

1. A `_pps_cmd(...)` helper that renders one `bench_pps` command.
2. A `dist_ratio_pps(...)` generator that creates the full experiment matrix.
3. A `get_experiment_labels(...)` branch for PPS labels.
4. A new `EMBEDDED_GENERATORS["dist_ratio_pps"]` registry entry.

`experiment_runner.py` remains unchanged as the generic execution engine.

## Generator Behavior

### Sweep Dimensions

The PPS experiment generator will iterate in this order:

`distribution × algorithm × cross_ratio`

This matches the intended runner behavior and keeps experiment ordering deterministic for labels and result collection.

### Distribution Semantics

PPS does not use a `skew_factor` flag. Instead, it uses `--zipf`.

The generator will map distribution names to PPS Zipf values as follows:

- `normal` -> `zipf = 0.0`
- `skew` -> `zipf = 0.9`

Default distributions:

- `["skew", "normal"]`

Default Zipf values:

- `zipf_for_skew_dist = 0.9`
- `zipf_for_normal_dist = 0.0`

These defaults live inside the generator so the behavior is explicit and local to PPS.

### Algorithm and Ratio Defaults

The generator should follow the same resolution pattern already used by the existing generators:

- validate algorithms with `_resolve_algorithms(...)`
- use the shared cross-ratio defaults when explicit values are not provided

This keeps PPS aligned with the current experiment model and CLI conventions.

## Command Construction

### New Helper

Add `_pps_cmd(...)` beside `_ycsb_cmd(...)` and `_tpcc_cmd(...)`.

Inputs:

- `node_id`
- `servers`
- `n_nodes`
- `cfg: AlgorithmConfig`
- `cross_ratio`
- `zipf`

The helper will build a command targeting:

- `${BIN_PREFIX}/bench_pps`

The command should include the shared cluster/runtime flags already used by the other helpers where relevant:

- `--logtostderr=1`
- `--id`
- `--servers`
- `--protocol`
- `--partition_num`
- `--threads`
- `--partitioner`

The command should include PPS workload flags:

- `--cross_ratio=<ratio>`
- `--zipf=<value>`

If Clay/Lion-family algorithms require the same router/migration/runtime flags in PPS that they use in existing helpers, that behavior should be expressed in `_pps_cmd(...)` with the same style as the current command builders. The implementation should prefer consistency with the existing algorithm configuration model over inventing PPS-only branching elsewhere.

## Labels and Result Naming

`get_experiment_labels(...)` must gain a `dist_ratio_pps` branch that mirrors the exact iteration order of the new generator.

Label format:

- `skew_Lion_cross50`
- `normal_Silo_cross20`

This ensures:

- result directory names remain meaningful
- labels stay aligned with generated commands
- debugging mismatched outputs stays straightforward

## Validation Rules

The PPS generator should validate:

- unknown algorithms using the existing `_resolve_algorithms(...)` path
- unknown distributions by allowing only `skew` and `normal`

Validation errors should be explicit and match the tone/style of current generator errors.

## Data Flow

The runtime flow remains:

1. `experiment_runner.py` parses CLI arguments.
2. The runner passes generator parameters into `generate_commands(...)`.
3. `dist_ratio_pps(...)` generates one command per experiment configuration for each node.
4. The runner starts the remote nodes and local node using the generated PPS commands.
5. The runner waits for the configured runtime.
6. The runner stops the benchmark processes and collects outputs using the existing result pipeline.

No new orchestration code path is needed.

## Error Handling

Expected failure modes:

- invalid algorithm name
- invalid distribution token
- empty or inconsistent generated command lists

The implementation should rely on existing generator validation patterns and avoid introducing PPS-specific orchestration error handling unless the current generic runner cannot surface a clear failure.

## Testing and Verification

Verification should focus on command generation and ordering rather than adding a new execution path.

Minimum checks:

1. Run the new generator in `--dry-run` mode.
2. Confirm the number of generated experiments equals:
   `len(distributions) × len(algorithms) × len(cross_ratios)`.
3. Confirm at least one `normal` command contains `--zipf=0.0`.
4. Confirm at least one `skew` command contains `--zipf=0.9`.
5. Confirm labels match command order.
6. Confirm `experiment_runner.py` can invoke `--generator dist_ratio_pps` without requiring any orchestration changes.

## Implementation Scope

Primary code changes are expected in:

- `revision_scripts/experiments.py`

Possible light-touch follow-up only if required by naming or usability:

- `revision_scripts/experiment_runner.py`

The preferred implementation keeps the CLI stable and avoids adding PPS-specific top-level options unless a clear need appears during implementation.

## Rationale

This approach fits the current architecture because:

- `revision_scripts` is the actual active runner system
- workload-specific behavior already lives in embedded generators
- the generic runner already knows how to execute any generator in the registry
- PPS already exists as a benchmark binary, so only the orchestration layer is missing

Compared with reviving legacy per-workload scripts, this design reduces duplication and keeps all modern experiment definitions in one place.

## Open Decisions Resolved

- Source of truth: `revision_scripts`, not `scripts/`
- PPS runner shape: embedded generator, not a standalone legacy script
- Experiment matrix: `distribution × algorithm × cross_ratio`
- PPS skew control: use `zipf`, not `skew_factor`
- Default PPS uniform behavior: `zipf = 0.0`
- Default skew distribution behavior in the runner: `zipf = 0.9`
