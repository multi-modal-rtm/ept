# Skill: ept-data

Detection, tracking, and the frozen feature cache.

- Sampling: T=32 frames uniformly, pooled into S=8 segments.
- Detector SCRFD (`buffalo_l`); fall back to YOLO person-class only if the Phase 1 QA gate fails.
- Tracker: `supervision` ByteTrack. No re-ID network — it would add cost to the line item we
  are trying to report honestly.
- Entity selection: top-E by (mean confidence × frame coverage), E_max=8. Identical code path in
  every condition.
- Absent (entity, segment) pairs get the `[ABSENT]` embedding and an attention mask entry.
  Never zero-fill silently.
- Feature layout: `[E_max, S, D]` float16 plus a boolean presence mask `[E_max, S]`.
- The A0 grid baseline is extracted from the **same** backbone in the same pass. Matched features
  are a correctness requirement, not an optimisation.
- OUC-CGE: exclude `low/view2572.mp4`, `low/view2531.mp4`. 771-clip video-only test protocol.
- MELD: align by `(dialogue_id, utterance_id)` from the label CSV, never by directory listing.
  Exclude `data/meld/bad_clips.txt`. Use face-embedding clustering, not ByteTrack — shot cuts.
- rclone the cache off-server as soon as it is complete.
