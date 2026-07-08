# Conservation Laws for Diffusion Models

This repository contains the runnable code artifact for the paper "Conservation
Laws for Diffusion Models". It includes the released synthetic Markov,
byte-level `text8`, and CIFAR-10 RGB8 image-token experiment paths.

Project site: <https://zivaharoni.github.io/conservation-laws-diffusion-models/>

For setup, data preparation, paper experiment commands, smoke checks, Slurm
launchers, evaluation commands, and tests, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## What This Code Reproduces

- Synthetic Markov experiments for causal CE, EXIT, I-MMSE, and QSC baselines.
- Markov channel-comparison runs at `K=128` for EXIT, QSC, and I-MMSE.
- Byte-level `text8` language-model experiments for causal CE, EXIT, I-MMSE,
  and QSC.
- CIFAR-10 RGB8 image-token experiments for EXIT, I-MMSE, and QSC with the
  repo-native UNet backbone.

The canonical paper commands use the explicit `paper_*` Hydra configs where
available. Neighboring configs without the `paper_` prefix are development or
lower-memory presets unless `REPRODUCIBILITY.md` states otherwise.

## Repository Layout

- `src/`: reusable models, objectives, channels, datasets, and training logic.
- `configs/`: Hydra configs for released experiment presets.
- `scripts/`: command-line entrypoints for training, evaluation, dataset
  bootstrap, reporting, and paper-artifact export.
- `experiments/`: Slurm launchers, reproducibility manifests, and shared
  experiment utilities.
- `tests/`: regression tests for the released code path.
- `docs/`: GitHub Pages landing page and lightweight web figure assets.

## Citation

If you use this code, please cite:

```bibtex
@misc{aharoni2026conservationlaws,
  title = {Conservation Laws for Diffusion Models},
  author = {Aharoni, Ziv and Pfister, Henry D.},
  year = {2026},
  note = {Code release},
  url = {https://github.com/zivaharoni/conservation-laws-diffusion-models}
}
```
