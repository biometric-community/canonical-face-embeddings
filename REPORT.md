# REPORT — Canonical Face Embeddings

**Protocol:** `full`

Lab twin: Softmax / CosFace / ArcFace ResNet-18 on CelebA; OLS linear and
Wahba–Kabsch rotation maps; LFW within- vs cross-CNN TAR@FAR=1e-2 (paper LFW pilot).

## Training

- Losses: softmax, cosface, arcface
- Train images: 141610
- Identities: 7124
- Enrollment images for maps: 1024

![Training](outputs/figures/full/train_loss.svg)

## Within-CNN LFW

| Model | Accuracy | EER | TAR@FAR1e-2 |
|-------|----------|-----|-------------|
| softmax | 61.05% | 38.95% | 3.10% |
| cosface | 72.83% | 27.17% | 16.00% |
| arcface | 50.48% | 49.52% | 3.17% |

## Cross-CNN summary (LFW)

| Mapping | Mean TAR@FAR1e-2 |
|---------|------------------|
| Within (mean) | 7.42% |
| Linear | 1.26% |
| Rotation | 1.16% |

![Cross](outputs/figures/full/cross_summary.svg)

![Linear matrix](outputs/figures/full/cross_linear_matrix.svg)

![Rotation matrix](outputs/figures/full/cross_rotation_matrix.svg)

## Notes

- Paper primary bench is IJB-C (10 commercial CNNs); we follow the LFW pilot with 3 CelebA-trained heads.
- See `DEVIATIONS.md`.
