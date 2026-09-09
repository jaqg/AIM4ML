"""
provenance.py — Per-invocation audit logging for the AIM4ML pipeline.

Each stage calls record_run() at startup.  The first call initializes
top-level metadata (git commit/dirty, RDKit/Python versions, hostname);
subsequent calls append stage records.  Append-only, so partial reruns
and parameter sweeps accumulate a full audit trail in one file.
"""

import json
import os
import sys
import subprocess
import datetime
from rdkit import rdBase

# Directory containing the stage scripts (parent of lib/).  Git commands run
# from here (finds the enclosing repo), and .git-commit/.git-dirty fallback
# files live here too — synced to the cluster via src/.
_CURATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_commit():
    """Return the git HEAD commit, or the .git-commit fallback, or None.

    The cluster has no git repo, so git fails there.  The .git-commit file
    (regenerated locally before syncing) carries the hash instead.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_CURATION_DIR, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        pass
    commit_file = os.path.join(_CURATION_DIR, ".git-commit")
    if os.path.exists(commit_file):
        with open(commit_file) as f:
            return f.read().strip()
    return None


def _git_dirty():
    """Return True if the working tree is dirty, .git-dirty fallback, or None."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_CURATION_DIR, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return bool(out)
    except Exception:
        pass
    dirty_file = os.path.join(_CURATION_DIR, ".git-dirty")
    if os.path.exists(dirty_file):
        return open(dirty_file).read().strip().lower() == "true"
    return None


def record_run(output_path, stage_name):
    """
    Append a stage record to BASE/provenance.json (atomically).

    Parameters
    ----------
    output_path : str
        The stage's output path.  For stage 0 this is a file
        (BASE/valid.sdf); for stages 1-10 it is a directory
        (BASE/batches/, BASE/stats/, ...).  The pipeline base dir is its
        parent directory, so all stages converge on the same file.
    stage_name : str
        E.g. "03_filter", "08_conformer_filter".
    """
    base_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, "provenance.json")

    # Load existing, or initialize top-level metadata
    if os.path.exists(path):
        with open(path) as f:
            prov = json.load(f)
    else:
        prov = {
            "pipeline": "AIM4ML-curation",
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "hostname": os.uname().nodename,
            "python_version": sys.version.split()[0],
            "rdkit_version": rdBase.rdkitVersion,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stages": [],
        }
        if prov.get("git_dirty"):
            print(f"WARNING: Git working tree has uncommitted changes. "
                  f"Commit hash: {prov['git_commit']}", file=sys.stderr)

    # Append stage record
    stage_record = {
        "stage": stage_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }

    # HPC environment variables (log if present)
    hpc_env = {}
    for key in ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NTASKS",
                "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES"]:
        val = os.environ.get(key)
        if val:
            hpc_env[key] = val
    if hpc_env:
        stage_record["hpc_env"] = hpc_env

    prov["stages"].append(stage_record)

    # Atomic write
    tmp = path + ".tmp." + str(os.getpid())
    with open(tmp, "w") as f:
        json.dump(prov, f, indent=2)
    os.replace(tmp, path)
