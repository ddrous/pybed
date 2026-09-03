# Changelog

## 0.2.0 - 2026-09-03

- Store designs as `Batch.x`, outcomes as `Batch.y`, and the complete pair as `Batch.o` or `Batch.obs`; retain `design` and `outcome` aliases.
- Add one-pass `infer_design`, `design_predict`, and `design_infer_predict` component calls.
- Add weighted particle clouds, score-function policy samples, candidate pools, changing location shapes, and replay.
- Rename the main data helpers to `Stream` and `Seeds`, with readable compatibility aliases.
- Vectorize non-adaptive rollouts and accept fixed design sequences.
- Support no-design inverse datasets and unbiased energy-score training.
- Add the Action-BED MNIST classification environment.
- Add energy-transport and Lightning/W&B DAD examples.
- Rewrite the README and technical report around the complete observation convention and all supported method families.

## 0.1.0 - 2026-09-03

- Introduce the composable `BED`, `Batch`, `Spec`, and prior contracts.
- Add location, MNIST image discovery, advection-diffusion, stochastic pendulum, CES, and death-process environments.
- Add repeatable random streams, finite synthetic epochs, offline datasets, and deterministic loaders.
- Add posterior, calibration, information, and source-set metrics.
- Add camera-ready visualizers and local experiment records.
- Add random-policy and jointly trained ActionBED examples.
- Add unit, gradient, reproducibility, serialization, packaging, and end-to-end tests.
