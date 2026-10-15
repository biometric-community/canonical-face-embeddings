"""Figures + REPORT.md for canonical face embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from canemb.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(models_hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, m in enumerate(models_hist):
        hist = m["history"]
        ax.plot(
            [h["epoch"] for h in hist],
            [h["loss"] for h in hist],
            color=MULTI_SERIES_COLORS[i % len(MULTI_SERIES_COLORS)],
            linewidth=LINEWIDTH_MAIN,
            label=m["loss"],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("CelebA FR heads (Softmax / CosFace / ArcFace)")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_cross(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    within = eval_res["within"]
    names = list(within.keys())
    # mean within TAR
    within_tar = [within[n]["tar@far0.01"] * 100 for n in names]
    # mean cross linear / rotation
    lin = list(eval_res["cross_linear"].values())
    rot = list(eval_res["cross_rotation"].values())
    mean_lin = float(np.mean([v["tar@far0.01"] for v in lin])) * 100 if lin else 0
    mean_rot = float(np.mean([v["tar@far0.01"] for v in rot])) * 100 if rot else 0
    mean_within = float(np.mean(within_tar))

    labels = ["Within (mean)", "Cross linear", "Cross rotation"]
    vals = [mean_within, mean_lin, mean_rot]
    ax.bar(labels, vals, color=list(MULTI_SERIES_COLORS), edgecolor="#002080")
    ax.set_ylabel("TAR @ FAR=1e-2 (%)")
    ax.set_title("LFW: within-CNN vs mapped cross-CNN")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_matrix(cross: dict, out: Path, title: str) -> None:
    apply_rcparams()
    keys = sorted(cross.keys())
    srcs = sorted({k.split("->")[0] for k in keys})
    tgts = sorted({k.split("->")[1] for k in keys})
    mat = np.full((len(srcs), len(tgts)), np.nan)
    for i, s in enumerate(srcs):
        for j, t in enumerate(tgts):
            if s == t:
                continue
            k = f"{s}->{t}"
            if k in cross:
                mat[i, j] = cross[k]["tar@far0.01"] * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(tgts)))
    ax.set_yticks(range(len(srcs)))
    ax.set_xticklabels(tgts, rotation=30, ha="right")
    ax.set_yticklabels(srcs)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(title)
    for i in range(len(srcs)):
        for j in range(len(tgts)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    within = eval_res.get("within", {})
    lin = eval_res.get("cross_linear", {})
    rot = eval_res.get("cross_rotation", {})
    mean_w = float(np.mean([v["tar@far0.01"] for v in within.values()])) if within else 0
    mean_l = float(np.mean([v["tar@far0.01"] for v in lin.values()])) if lin else 0
    mean_r = float(np.mean([v["tar@far0.01"] for v in rot.values()])) if rot else 0

    lines = [
        "# REPORT — Canonical Face Embeddings",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin: Softmax / CosFace / ArcFace ResNet-18 on CelebA; OLS linear and",
        "Wahba–Kabsch rotation maps; LFW within- vs cross-CNN TAR@FAR=1e-2 (paper LFW pilot).",
        "",
        "## Training",
        "",
        f"- Losses: {', '.join(train_s.get('losses', []))}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Enrollment images for maps: {eval_res.get('n_enroll')}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Within-CNN LFW",
        "",
        "| Model | Accuracy | EER | TAR@FAR1e-2 |",
        "|-------|----------|-----|-------------|",
    ]
    for k, v in within.items():
        lines.append(
            f"| {k} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['tar@far0.01']*100:.2f}% |"
        )
    lines += [
        "",
        "## Cross-CNN summary (LFW)",
        "",
        f"| Mapping | Mean TAR@FAR1e-2 |",
        f"|---------|------------------|",
        f"| Within (mean) | {mean_w*100:.2f}% |",
        f"| Linear | {mean_l*100:.2f}% |",
        f"| Rotation | {mean_r*100:.2f}% |",
        "",
        f"![Cross]({fig_dir.as_posix()}/cross_summary.svg)",
        "",
        f"![Linear matrix]({fig_dir.as_posix()}/cross_linear_matrix.svg)",
        "",
        f"![Rotation matrix]({fig_dir.as_posix()}/cross_rotation_matrix.svg)",
        "",
        "## Notes",
        "",
        "- Paper primary bench is IJB-C (10 commercial CNNs); we follow the LFW pilot with 3 CelebA-trained heads.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["models"], fig_dir / "train_loss.svg")
    plot_cross(eval_res, fig_dir / "cross_summary.svg")
    plot_matrix(eval_res["cross_linear"], fig_dir / "cross_linear_matrix.svg", "Linear map TAR@1e-2")
    plot_matrix(eval_res["cross_rotation"], fig_dir / "cross_rotation_matrix.svg", "Rotation map TAR@1e-2")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
