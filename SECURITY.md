# Security policy

## Reporting a vulnerability

Please open a private security advisory on GitHub (Security tab, "Report a vulnerability") instead of a public issue.

## Hardening in this repository

* Dependencies are pinned with hashes (`requirements*.txt`, compiled by uv) and installed with `--require-hashes`.
* CI runs `pip-audit`, a Trivy image scan and a Trivy misconfiguration scan, weekly and on every push.
* Containers run as non-root, with `cap_drop: ALL`, `no-new-privileges`, a read-only root filesystem, PID limits and memory limits.
* App and LLM server sit on an internal Docker network with no internet egress. Only nginx is attached to an external network and it publishes to `127.0.0.1` only.
* The LLM cannot execute code or reach the network: it can only pick from six validated, read-only analysis functions.
* GitHub Actions are pinned to commit SHAs and run with read-only permissions.

## Scope

This is a local analytics tool for a public research dataset. It has no authentication layer; keep the published port bound to localhost or put an authenticating reverse proxy in front of it.
