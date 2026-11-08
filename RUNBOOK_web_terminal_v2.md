# RUNBOOK - restarting the pod for A2-v2

Web terminal is used for ONE thing: restoring SSH access after a restart.
Everything else runs over exposed-TCP SSH from the laptop, as with A1 and A2-v1.

**Why the web terminal is needed at all.** RunPod injects SSH keys only at pod
CREATION, and `/root/.ssh/` lives on the container filesystem, not the persistent
volume - so every restart wipes `authorized_keys`. This bit us twice already
(2026-09-02 and again 2026-09-03 16:06).

**State as of 2026-09-03 15:00 EDT:** pod STOPPED (last address
`root@69.30.85.145 -p 22078` refuses connections). `/workspace` persists, and
holds the project, `data/`, the ~15 GB weight cache, `runs/`, and
`/workspace/.ssh_backup/authorized_keys` (created 2026-09-03 16:08 UTC).

---

## 1. Start the pod

RunPod console -> the existing pod -> **Start**. Do not create a new one: a new
pod loses `/workspace`, which means re-uploading 34 MB of frozen `data/` and
re-downloading ~15 GB of weights.

## 2. Web terminal - restore SSH access (one line)

Connect -> **Start Web Terminal**, then paste:

```bash
mkdir -p ~/.ssh && cp /workspace/.ssh_backup/authorized_keys ~/.ssh/ && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && echo RESTORED && wc -l ~/.ssh/authorized_keys
```

If `/workspace/.ssh_backup/` is somehow gone, fall back to appending the public
key directly (this is the public half; the private key never leaves the laptop):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHEoX7EPRWdqOCL/z2z39hbGHADV/D8kXrHuZt77B3l5 rachel.calcott9@gmail.com' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
mkdir -p /workspace/.ssh_backup && cp ~/.ssh/authorized_keys /workspace/.ssh_backup/
```

## 3. Sanity-check the volume survived (same web terminal, optional)

```bash
ls -d /workspace/self-interp-openweight && \
wc -l /workspace/self-interp-openweight/data/train/A2/train_sft.jsonl && \
ls /workspace/self-interp-openweight/runs/ && \
du -sh /workspace/hf_cache 2>/dev/null
```

## 4. Paste the new SSH command

RunPod reassigns the exposed TCP IP and port on restart, so the old
`69.30.85.145:22078` is dead. Copy the full command from the pod's Connect panel:

```
ssh root@<new-ip> -p <new-port> -i ~/.ssh/id_ed25519
```

---

## From there it is all SSH, no web terminal

Once the address is pasted:

```bash
# 1. sync the v2 code (v2 files all postdate the pod's 13:04 tree)
rsync -av -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
  --exclude .venv --exclude runs --exclude __pycache__ --exclude .pytest_cache \
  ~/Desktop/Coordination/self-interp-openweight/ \
  root@<host>:/workspace/self-interp-openweight/

# 2. pre-flight on the pod (free): 71 tests, v1-vs-v2 config diff, choice_only masking
# 3. scripts/run_cloud_a2_v2.sh under nohup/tmux so it survives a dropped SSH session
# 4. stream the gates, then retrieve and hash-verify artifacts
```

`data/` transfers in that rsync but is unchanged since Aug 31, so it is a no-op
diff. The sealed splits (`policy_test`, `self_pred`, `relational`) are covered by
the in-run reserved-split guard, which reports every file read.

**Billing, unchanged and still yours to clear:** the script blocks on a typed
`I CONFIRM` (auto-recharge OFF, balance <= $25). No watchdog, no `RUNPOD_API_KEY`
on the pod, no self-termination - per the 2026-09-02 decision. Stopping the pod
does not delete the volume; deleting the volume is what ends the storage charge.
