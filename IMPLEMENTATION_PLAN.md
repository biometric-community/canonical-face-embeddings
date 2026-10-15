# IMPLEMENTATION_PLAN — Canonical Face Embeddings

## Goal
Reproduce the paper's claim that diverse FR embedding spaces are related by linear / rotation maps, using the LFW pilot protocol under the 5 GiB gate.

## Figure inventory
| Paper fig | Our artifact |
|-----------|--------------|
| Fig. 5–6 LFW cross-map | `cross_summary.svg`, `cross_*_matrix.svg` |
| Training (lab) | `train_loss.svg` |

## Size gate
celeba + lfw + calfw + cplfw ≈ 4.1 GiB → full_train.
