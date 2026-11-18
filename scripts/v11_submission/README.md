# v11 manuscript and local handoff verification

Run from the code root with system Python. Every output path must be new.

- `prepare_handoff.py --output reports/<new_attempt>` checks exact corrected
  headline rows, split integrity, saved held-out hit means, protocol separators,
  historical table/algorithm/figure preservation and model/data hashes. Captures
  the actual environment and its dependency conflicts. Does not install or train.
- `verify_release.py --output reports/<new_attempt>` checks PDF text/bounds and
  paper endpoint byte identity, renders pages to the master's new `qa/`, runs
  the existing API/replay tests and checks the demo archive plan. A second run
  needs a new QA location/copy, not an overwrite of this delivered review.
- `seal_handoff.py --output reports/<new_attempt>` was used once after visual
  review. It rechecks historical provenance, seals the resolved environment
  lock from the saved pip report and hashes the final deliverables. It refuses
  to replace that already sealed lock on a second run.

The manuscript's exclusive-creation source transformer is
`../research-paper-v11-evaluation/scripts/prepare_revision.py`. It retains
before/after prose in `changes.json`. Compiler: tectonic. Models, raw results,
figure data and old manuscripts are inputs only.

Completed receipts are under `reports/v11_submission_20261116_02/`,
`v11_release_checks_20261116_01/`, `v11_environment_resolution_20261116_01/`
and `v11_handoff_seal_20261116_01/`. The first empty/failed verification attempt
has its own explanatory README and is not a completed result.
