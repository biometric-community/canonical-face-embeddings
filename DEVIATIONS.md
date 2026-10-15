# DEVIATIONS — Canonical Face Embeddings

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Models | 10 commercial CNNs (MobileNet…R100, diverse train sets) | 3× ResNet-18 (Softmax/CosFace/ArcFace) on CelebA | Size gate; no IJB-C / Glint payloads |
| D2 | Primary bench | IJB-C 1:1 | LFW (+ CALFW/CPLFW within CosFace) | Paper LFW pilot; IJB-C stub only |
| D3 | Enrollment | 11,856 IJB-C images | CelebA val subset (`n_enroll`) | Available open data |
| D4 | Emb dim | Mostly 512-D | 128-D (smoke 64-D) | Local compute |
