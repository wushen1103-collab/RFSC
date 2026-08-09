# Final remote system snapshot

- Host kernel: Linux 6.8.0-111-generic, x86_64 (Ubuntu 22.04 family)
- Python: 3.11.15
- CPU: 2 x AMD EPYC 9654, 96 physical cores/socket, 384 logical CPUs
- Experiment affinity: logical CPUs 0--63; at least 30 physical cores were
  left outside the experiment allocation.
- GPU: no GPU was used for the Delta-LightGBM, Hodge mechanism, or incomplete-
  flow experiments. Earlier neural matrices record their own GPU/runtime rows.
- Final packages: `environment/requirements-final.lock`
- China mirror used for acquisition:
  `https://mirrors.aliyun.com/pypi/simple`. The mirror changes only transport;
  canonical package names and versions are frozen in the lock file.

