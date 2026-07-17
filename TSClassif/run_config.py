"""Pure parsing helpers for selecting reproducible experiment subsets."""


def parse_scenario(value):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError("--scenario must be SRC,TGT, for example 18,14")
    return [(parts[0], parts[1])]


def parse_run_ids(value):
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if not parts or not all(parts):
        raise ValueError("--run_ids must be a comma-separated list of non-negative integers")
    try:
        run_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--run_ids must contain integers only") from exc
    if any(run_id < 0 for run_id in run_ids):
        raise ValueError("--run_ids cannot contain negative integers")
    return list(dict.fromkeys(run_ids))


def resolve_run_ids(value, num_runs):
    if num_runs < 1:
        raise ValueError("--num_runs must be at least 1")
    explicit = parse_run_ids(value)
    return list(range(num_runs)) if explicit is None else explicit

