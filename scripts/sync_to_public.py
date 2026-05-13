#!/usr/bin/env python3
"""Sync whitelisted files from the private repo to the public repo.

Reads ``whitelist.txt`` in the repo root, copies only those files
to a temp directory, commits with the current user's git identity,
and pushes to the public remote.

Usage::

    # Dry run (no push)
    uv run scripts/sync_to_public.py --dry-run

    # Push to public repo
    uv run scripts/sync_to_public.py

Requirements:
    - ``git`` on PATH
    - ``PAT`` env var set with a GitHub personal access token that
      has write access to ``Im-Busy/investment_trying_lab``.

Environment:
    GIT_AUTHOR_NAME / GIT_AUTHOR_EMAIL  -- override commit author
    PAT                                 -- GitHub personal access token (required)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WHITELIST_FILE = REPO_ROOT / "whitelist.txt"
PUBLIC_OWNER = "Im-Busy"
PUBLIC_REPO = "investment_trying_lab"
PUBLIC_BRANCH = "main"


def _read_whitelist(path: Path) -> list[str]:
    """Parse the whitelist file, returning relative paths."""
    entries: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
    return entries


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command and return stdout; raise on failure and print stderr."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync files to the public repo")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the sync directory but do not push",
    )
    parser.add_argument(
        "--remote",
        default=f"https://github.com/{PUBLIC_OWNER}/{PUBLIC_REPO}.git",
        help="Public repo remote URL (default uses PAT env var)",
    )
    args = parser.parse_args()

    pat = os.environ.get("PAT", "")
    if not pat and not args.dry_run:
        print("ERROR: PAT env var is required for pushing", file=sys.stderr)
        print("  export PAT=ghp_...", file=sys.stderr)
        sys.exit(1)

    remote_url = f"https://x-access-token:{pat}@github.com/{PUBLIC_OWNER}/{PUBLIC_REPO}.git"

    # ---- read whitelist ----
    if not WHITELIST_FILE.exists():
        print(f"ERROR: whitelist not found at {WHITELIST_FILE}", file=sys.stderr)
        sys.exit(1)

    entries = _read_whitelist(WHITELIST_FILE)
    print(f"Loaded {len(entries)} whitelist entries")

    # ---- copy into temp dir ----
    tmp = Path(tempfile.mkdtemp(prefix="public_sync_"))
    print(f"Staging directory: {tmp}")

    copied = 0
    for entry in entries:
        src = REPO_ROOT / entry
        if not src.exists():
            print(f"  [WARN] Missing: {entry}")
            continue
        dst = tmp / entry
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"  Copied: {entry}")
        copied += 1

    print(f"Copied {copied} of {len(entries)} entries")

    # ---- init git and commit ----
    _run(["git", "init"], cwd=tmp)
    _run(["git", "checkout", "-b", PUBLIC_BRANCH], cwd=tmp)

    author_name = os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("USER", "unknown")
    author_email = os.environ.get("GIT_AUTHOR_EMAIL") or f"{author_name}@users.noreply.github.com"

    _run(["git", "config", "user.name", author_name], cwd=tmp)
    _run(["git", "config", "user.email", author_email], cwd=tmp)

    _run(["git", "add", "-A"], cwd=tmp)

    # Check if there are changes to commit
    diff_result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=tmp,
        capture_output=True,
        check=False,
    )
    if diff_result.returncode == 0:
        print("No changes to sync. Public repo is up to date.")
        return

    _run(["git", "commit", "-m", "Sync from private repo"], cwd=tmp)

    if args.dry_run:
        print("\nDRY RUN -- not pushing. Staging at:", tmp)
        print(_run(["git", "log", "--oneline", "-1"], cwd=tmp))
        print("Files staged:")
        print(_run(["git", "ls-tree", "--name-only", "HEAD"], cwd=tmp))
        return

    # ---- push ----
    _run(["git", "remote", "add", "public", remote_url], cwd=tmp)
    _run(["git", "push", "public", PUBLIC_BRANCH, "--force"], cwd=tmp)

    print(f"\nSynced successfully to {PUBLIC_OWNER}/{PUBLIC_REPO}")
    print("Your commits are now visible on the public repo.")


if __name__ == "__main__":
    main()
