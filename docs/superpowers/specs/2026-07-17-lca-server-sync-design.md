# LCA Server Sync Design

## Goal

Upload the local LCA code to `user@10.150.10.38:/data/user/LCA` without nesting another `LCA` directory and without replacing server-side datasets or experiment artifacts.

## Design

The repository-root `sync_to_server.sh` creates a temporary tar archive whose top-level entry is `LCA`. It excludes Git metadata, IDE/cache files, Python bytecode, `TSClassif/dataset`, result directories, logs, checkpoints, and common temporary files. The archive is copied to `/data/user`, then extracted with `/data/user` as the parent directory, overlaying code at the exact `/data/user/LCA` path while preserving excluded server content.

The SSH user, host, and target directory are configurable with environment variables; defaults match the supplied server. Strict Bash error handling and an exit trap remove the local temporary archive. The remote archive is removed only after successful extraction.

## Verification

- A standard-library contract test verifies the exact destination, exclusions, transport commands, and absence of recursive directory scp.
- Git for Windows Bash performs `bash -n sync_to_server.sh`.
- The script is not executed automatically because that would mutate the remote server.

