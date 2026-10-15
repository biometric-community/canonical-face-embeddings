# Canonical Face Embeddings

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of McNeely-White et al. (TBIOM 2022): Softmax / CosFace / ArcFace ResNet-18 on CelebA, OLS linear and Wahba–Kabsch rotation maps, LFW within- vs cross-CNN verification.

**Repository:** https://github.com/biometric-community/canonical-face-embeddings

## Setup

```bash
bash scripts/setup_env.sh
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
