"""MELD 7-class emotion labels — secondary calibration endpoint only
(docs/DECISION_RULES.md, 2026-08-15 amendment). NOT part of the primary
3-class sentiment endpoint or the H1/H2 branch decision. Reads directly from
the same raw label CSVs detect_cluster_meld.py uses, keyed by the same
clip_id convention (dia{N}_utt{M}) so it joins cleanly against the existing
cache/tracks/meld/<split>/*.json manifest without needing a new tokenization
or feature-extraction pass — emotion is a label swap, not a different input.
"""
import csv
import os

from ept.tokenization.detect_cluster_meld import LABEL_CSVS, MELD_ROOT

EMOTION_TO_INT = {
    "anger": 0, "disgust": 1, "fear": 2, "joy": 3,
    "neutral": 4, "sadness": 5, "surprise": 6,
}
NUM_EMOTION_CLASSES = 7


def load_emotion_labels(split):
    """split in {"train","dev","test"} -> {clip_id: emotion_int}. Callers that
    must not touch test (this calibration endpoint included) simply never
    pass split="test"; nothing here enforces that restriction itself, since
    this is a label lookup, not a Dataset class."""
    labels = {}
    with open(os.path.join(MELD_ROOT, "labels", LABEL_CSVS[split])) as f:
        for row in csv.DictReader(f):
            clip_id = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            labels[clip_id] = EMOTION_TO_INT[row["Emotion"]]
    return labels
