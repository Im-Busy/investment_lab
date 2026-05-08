---
description: Audit and clean up the project file system. Flags misplaced files, duplicate directories, stray artifacts, and produces a safe migration plan. Executes moves/renames with strict safety protocols.
---

# Housekeeper — File System Audit & Cleanup

Run an audit of the project file system, classify issues, and produce a migration plan for safe execution.

**Safety guarantees:**
- git-tracked files: uses `git mv` (atomic, revertible via `git checkout`)
- Non-git files: uses `robocopy /MOV` (copy-then-delete per-file, with retry logic)
- Never deletes a file before verifying the destination exists and matches
- All moves are one-step atomic operations (no separate copy + delete)

## Usage

```
/housekeeper audit     # Scan the tree and produce an issue report (no changes)
/housekeeper plan      # Produce a migration plan for approval
/housekeeper execute   # Execute the approved migration plan
/housekeeper full      # Audit -> Plan -> wait for approval -> Execute
```

## Robocopy Quick Reference

The agent uses `robocopy` on Windows for safe file moves when `git mv` is not applicable:

| Flag | Purpose |
|------|---------|
| `/MOV` | Move files (copy then delete source — per-file atomic) |
| `/E` | Include subdirectories (including empty) |
| `/R:3` | Retry 3 times on failure |
| `/W:5` | Wait 5 seconds between retries |
| `/NP` | No progress display |
| `/LOG:file` | Write log to file for post-op verification |

### Single file move:
```
robocopy "C:\source\dir" "C:\dest\dir" "filename.py" /MOV /R:3 /W:5 /NP /LOG:move.log
```

### Entire directory move:
```
robocopy "C:\source\parent" "C:\dest\parent" /E /MOV /R:3 /W:5 /NP /LOG:move.log
```

### Verify after every robocopy:
```
Find-String "FAILED" move.log
```
Must show zero FAILED files and zero FAILED directories before proceeding.
