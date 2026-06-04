"""Resilience helpers for long sweeps — checkpoint/resume + additive git push.

A long CPU sweep can outlive the ephemeral container. To avoid losing hours of
compute on a container reclaim, sweeps should:

  1. Append results to a CSV and, on restart, SKIP cells already present
     (``CheckpointedCSV`` — resume).
  2. Periodically commit+push that CSV (force-added past ``.gitignore``) so a
     freshly-cloned container can resume from the pushed file (``git_checkpoint``).

Both are deliberately non-fatal: if git is offline the sweep keeps running and
just retries at the next checkpoint. Commits are verified server-side on push, so
no local signing setup is required.
"""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path


class CheckpointedCSV:
    """Append-only results CSV that knows which (key) rows are already done.

    Parameters
    ----------
    path:        CSV file path.
    fieldnames:  full column order.
    key_fields:  subset identifying a unique cell (e.g. model,d_model,d_state,seed).
    """

    def __init__(self, path: str | Path, fieldnames: list[str], key_fields: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.key_fields = key_fields
        self._done: set[tuple[str, ...]] = set()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path, newline="") as fh:
                for row in csv.DictReader(fh):
                    self._done.add(self._key(row))
        else:
            with open(self.path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    def _key(self, row: dict) -> tuple[str, ...]:
        return tuple(str(row[k]) for k in self.key_fields)

    def is_done(self, key: dict) -> bool:
        return self._key(key) in self._done

    def n_done(self) -> int:
        return len(self._done)

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.fieldnames).writerow(row)
            fh.flush()
        self._done.add(self._key(row))


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def git_checkpoint(paths: list[str | Path], message: str, push: bool = True,
                   retries: int = 4) -> bool:
    """Force-add ``paths``, commit, and push the current branch. Non-fatal.

    Returns True on a successful (commit [+ push]) cycle, False if it skipped or
    failed. Failures only print a warning so the sweep can continue.
    """
    try:
        _git("rev-parse", "--is-inside-work-tree")
    except Exception:
        print("[checkpoint] not a git repo — skipping", flush=True)
        return False

    try:
        _git("add", "-f", *[str(p) for p in paths])
        # nothing staged -> nothing to do
        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            return False
        _git("commit", "-m", message)
    except subprocess.CalledProcessError as e:
        print(f"[checkpoint] commit failed: {e.stderr.strip()}", flush=True)
        return False

    if not push:
        return True

    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    delay = 2
    for attempt in range(1, retries + 1):
        res = _git("push", "-u", "origin", branch, check=False)
        if res.returncode == 0:
            print(f"[checkpoint] pushed: {message}", flush=True)
            return True
        print(f"[checkpoint] push attempt {attempt}/{retries} failed: "
              f"{res.stderr.strip()[:200]}", flush=True)
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    print("[checkpoint] push gave up (committed locally, will retry next checkpoint)",
          flush=True)
    return False
