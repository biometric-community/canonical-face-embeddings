"""Within-CNN and cross-CNN (linear / rotation) verification on LFW."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from canemb.data.celeba import CelebAIdDataset, resolve_celeba_root
from canemb.data.pairs import build_calfw, build_cplfw, build_lfw
from canemb.mapping.maps import apply_map, fit_linear, fit_rotation
from canemb.models.fr_net import FaceEmbedder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer, best_t = 0.0, 1.0, 0.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer, best_t


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float = 1e-2) -> float:
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    t = float(np.quantile(neg, 1.0 - far))
    return float((pos >= t).mean())


def load_model(ckpt_path: Path, device: torch.device) -> FaceEmbedder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FaceEmbedder(
        int(ckpt["num_classes"]),
        emb_dim=int(ckpt.get("emb_dim", 128)),
        pretrained=False,
        loss=str(ckpt.get("loss", "softmax")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def embed_loader(model, loader, device) -> np.ndarray:
    outs = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        outs.append(model.embed(x.to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def pair_scores(model_a, model_b, ds, device, bs, nw, map_m=None) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    scores, labels = [], []
    for a, b, y in loader:
        ea = model_a.embed(a.to(device))
        if map_m is not None:
            ea = apply_map(ea, map_m)
        eb = model_b.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    losses: List[str] = list(cfg["model"].get("losses", ["softmax", "cosface", "arcface"]))
    models = {name: load_model(ckpt_dir / f"{name}.pt", device) for name in losses}

    image_size = int(cfg["model"].get("image_size", 112))
    max_pairs = cfg["eval"].get("max_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    bs = int(cfg["eval"].get("batch_size", 64))
    nw = int(cfg["train"].get("num_workers", 4))
    n_enroll = int(cfg["eval"].get("n_enroll", 1024))

    # Enrollment embeddings from CelebA val (open-set-ish relative to LFW)
    celeba = resolve_celeba_root(cfg)
    enroll_ds = CelebAIdDataset(
        celeba,
        split="val",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=n_enroll,
        seed=int(cfg["train"].get("seed", 42)),
        train=False,
    )
    enroll_loader = DataLoader(enroll_ds, batch_size=bs, shuffle=False, num_workers=nw)
    enroll_emb = {name: embed_loader(models[name], enroll_loader, device) for name in losses}

    maps: Dict[str, Dict[str, np.ndarray]] = {}
    for src in losses:
        for tgt in losses:
            if src == tgt:
                continue
            key = f"{src}->{tgt}"
            maps[key] = {
                "linear": fit_linear(enroll_emb[src], enroll_emb[tgt]),
                "rotation": fit_rotation(enroll_emb[src], enroll_emb[tgt]),
            }

    lfw = build_lfw(cfg, image_size, max_pairs)
    results = {"within": {}, "cross_linear": {}, "cross_rotation": {}, "n_enroll": len(enroll_ds)}

    for name in losses:
        s, y = pair_scores(models[name], models[name], lfw, device, bs, nw)
        acc, eer, _ = best_acc_eer(s, y)
        tar = tar_at_far(s, y, 1e-2)
        results["within"][name] = {
            "accuracy": acc,
            "eer": eer,
            "tar@far0.01": tar,
            "n_pairs": int(len(y)),
        }
        print(f"[within:{name}] acc={acc:.4f} eer={eer:.4f} tar@1e-2={tar:.4f}")

    for src in losses:
        for tgt in losses:
            if src == tgt:
                continue
            key = f"{src}->{tgt}"
            for kind in ("linear", "rotation"):
                s, y = pair_scores(
                    models[src], models[tgt], lfw, device, bs, nw, map_m=maps[key][kind]
                )
                acc, eer, _ = best_acc_eer(s, y)
                tar = tar_at_far(s, y, 1e-2)
                bucket = results[f"cross_{kind}"]
                bucket[key] = {
                    "accuracy": acc,
                    "eer": eer,
                    "tar@far0.01": tar,
                    "n_pairs": int(len(y)),
                }
                print(f"[cross:{kind}:{key}] acc={acc:.4f} tar@1e-2={tar:.4f}")

    # Optional CALFW/CPLFW within cosface only (compact)
    extra = {}
    for tag, builder in (("calfw", build_calfw), ("cplfw", build_cplfw)):
        try:
            ds = builder(cfg, image_size, max_pairs)
            s, y = pair_scores(models["cosface"], models["cosface"], ds, device, bs, nw)
            acc, eer, _ = best_acc_eer(s, y)
            extra[tag] = {"accuracy": acc, "eer": eer, "tar@far0.01": tar_at_far(s, y, 1e-2)}
            print(f"[within:cosface:{tag}] acc={acc:.4f}")
        except Exception as e:
            extra[tag] = {"error": str(e)}
    results["extra_within_cosface"] = extra

    # Persist maps as lists for reproducibility (small)
    maps_ser = {
        k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in maps.items()
    }
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval.json").write_text(json.dumps(results, indent=2) + "\n")
    (log_dir / "maps.json").write_text(json.dumps({"n_enroll": len(enroll_ds), "keys": list(maps)}) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
