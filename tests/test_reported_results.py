"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def within_accuracy(report: str, model: str) -> float:
    match = re.search(rf"\| {model} \| ([\d.]+)% \|", report)
    assert match, f"{model} row missing from REPORT.md"
    return float(match.group(1))


def summary(report: str, name: str) -> float:
    match = re.search(rf"\| {re.escape(name)} \| ([\d.]+)% \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return float(match.group(1))


@pytest.mark.parametrize("model", ["softmax", "cosface", "arcface"])
def test_each_model_is_clearly_above_chance(report, model):
    assert within_accuracy(report, model) >= 60.0


@pytest.mark.parametrize("model", ["softmax", "cosface", "arcface"])
def test_lfw_accuracy_reaches_the_pilot_experiment(report, targets, model):
    floor = targets["lfw_lowest_accuracy"] - targets["tolerance_pp"]
    assert within_accuracy(report, model) >= floor


@pytest.mark.parametrize("mapping", ["Linear", "Rotation"])
def test_cross_cnn_stays_close_to_within_cnn(report, mapping):
    """Fig. 5: mapped accuracy has a minimal drop; FAR 1e-2 chance level is 1%."""
    assert summary(report, mapping) >= 0.5 * summary(report, "Within (mean)")
