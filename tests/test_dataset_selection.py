"""Locks src/ept/train/train.py's dataset-selection safety net (2026-08-15
gate: "which script produced the 56 MELD Phase 3 runs" provenance audit).
Two things must hold, together, for the eventual one-shot test evaluation to
be trustworthy: (1) configs/train.yaml has no default dataset -- an
invocation that forgets `dataset=...` must fail loudly, not silently train on
whichever dataset happens to be first in a registry; (2) the dataset object
actually constructed must match what the config named, checked against both
its own declared identity and the cache path it actually read from.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from ept.train.dataset import OUCCGEDataset
from ept.train.dataset_meld import MELDDataset
from ept.train.train import DATASET_CLASSES, assert_dataset_matches_config

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_PRESENT = (
    os.path.isdir("/home/devops/ept/cache/features/ouccge/val")
    and os.path.isdir("/home/devops/ept/cache/features/meld/dev")
)


def test_train_yaml_has_no_default_dataset():
    """dataset: ??? in configs/train.yaml -- omitting dataset=... on the CLI
    must fail at config resolution, not fall back to a hardcoded class."""
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", "train.yaml"))
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.dataset


def test_dataset_registry_covers_both_datasets():
    assert set(DATASET_CLASSES) == {"ouccge", "meld"}
    assert DATASET_CLASSES["ouccge"] is OUCCGEDataset
    assert DATASET_CLASSES["meld"] is MELDDataset


@pytest.mark.skipif(not CACHE_PRESENT, reason="requires the real feature cache on disk")
def test_matching_dataset_passes():
    ouccge_ds = OUCCGEDataset("val", "A5", run_seed=42)
    meld_ds = MELDDataset("dev", "A5", run_seed=42)
    assert_dataset_matches_config("ouccge", ouccge_ds)  # must not raise
    assert_dataset_matches_config("meld", meld_ds)  # must not raise


@pytest.mark.skipif(not CACHE_PRESENT, reason="requires the real feature cache on disk")
def test_mismatched_dataset_fails_loudly():
    """The exact failure mode this gate exists to prevent: a config that
    names one dataset while the code path actually loaded another must not
    pass silently."""
    ouccge_ds = OUCCGEDataset("val", "A5", run_seed=42)
    meld_ds = MELDDataset("dev", "A5", run_seed=42)

    with pytest.raises(AssertionError):
        assert_dataset_matches_config("meld", ouccge_ds)
    with pytest.raises(AssertionError):
        assert_dataset_matches_config("ouccge", meld_ds)
