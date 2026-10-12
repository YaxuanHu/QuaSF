# Release verification, 2026-10-12

- `unit_tests.log`: eight source and numerical tests, all pass.
- `../CHECKPOINT_PARITY_20261012.json`: strict restoration and exact 16-tensor CPU forward equality against the original source, one validation protein.
- `standalone_smoke.json`: strict load and finite forward output using only copied release assets.
- `matched_region.json`: recomputation from saved validation/test caches, including both QuaSF and Region MLP.
- `table1.json`: localization rescoring of all 9,015 proteins from saved QuaSF predictions.
- `table2.json`: heterogeneous recovery rescoring from saved QuaSF predictions.

No new full-split GPU inference or training was performed for this release. Recorded paper results and fresh cache rescoring are distinguished from the CPU smoke test.
