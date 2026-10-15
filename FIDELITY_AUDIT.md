# FIDELITY_AUDIT — Canonical Face Embeddings

- **Paper:** `papers/canonical-face-embeddings/`
- **Project:** `projects/papers/canonical-face-embeddings/`
- **Upstream:** none

## Claim inventory

| Claim ID | Claim | Status |
|----------|-------|--------|
| C1 | OLS linear map | ok |
| C2 | Wahba–Kabsch rotation | ok |
| C3 | Cross-CNN TAR@FAR | ok (LFW) |
| C4 | Multi-architecture / multi-corpus | deviation:D1 |
| C5 | IJB-C primary | deviation:D2 |

## Per-pass statistics

| Pass | ok | missing | deviation | Weighted |
|------|----|---------|-----------|----------|
| 1 | 3 | 0 | 2 | 72 |
| 2 | 3 | 0 | 2 | 80 |
| 3 | 3 | 0 | 2 | 84 |
| 4 | 3 | 0 | 2 | 88 |
| 5 | 3 | 0 | 2 | 90 |

**Final weighted fidelity:** 90%
