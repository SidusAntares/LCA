# LCA Server Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a safe code-only upload script targeting `/data/user/LCA`.

**Architecture:** Package the repository from its parent so the archive contains one `LCA` root, copy it to `/data/user`, and extract there. Exclusions preserve server data and generated results.

**Tech Stack:** Bash, tar, scp, ssh, Python unittest contract test.

---

### Task 1: Add and verify the sync script

**Files:**
- Create: `sync_to_server.sh`
- Modify: `TSClassif/tests/test_offline_har_contract.py`
- Modify: `docs/LCA_RUN_AUDIT.md`

- [ ] Add a failing contract test for destination, exclusions, and transport behavior.
- [ ] Run the test and confirm failure because `sync_to_server.sh` is absent.
- [ ] Implement the tar/scp/ssh script with cleanup trap and code-only exclusions.
- [ ] Run the contract suite and confirm it passes.
- [ ] Run `bash -n sync_to_server.sh` and `git diff --check`.

