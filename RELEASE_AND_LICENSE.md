# Release and license boundary

This file is an engineering release note, not legal advice.

## Repository-owned code

No top-level license has been selected in this release. The code is published
for transparency and review; a reuse license should be added only after author
and institutional approval.

## Third-party resources

The repository does not vendor large third-party datasets, upstream source
checkouts, released prediction matrices, or row-level derived prediction files.
Instead, it provides configuration files, scripts, hashes, aggregate tables,
and instructions for regenerating those artifacts after the user obtains the
corresponding upstream resources.

## DeepDelta boundary

The pinned upstream DeepDelta commit is
`cd9b131559d54be639fcb9455e4c057a3dfd2c6b`. Its license should be reviewed
before any downstream redistribution. The upstream checkout is intentionally
excluded from this repository; use `external/deepdelta.lock.json` and
`REPRODUCING.md` to verify the exact source.

Aggregate tables and summary statistics are included for auditability. Row-level
upstream predictions and derived potentials should be regenerated locally unless
separate redistribution clearance is obtained.
