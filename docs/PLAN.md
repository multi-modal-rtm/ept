# EPT — Entity-Persistent Tokenization for Video Understanding
## Implementation plan & pre-registration — ICFNDS 2026 (ACM, İzmir, Nov 4–6 2026)

**Deadline of record:** **1 September 2026** (confirmed on the Sparcly submission portal, all tracks).
**Target track:** **Internet of Things (IoT)** — see §10.
**Format:** `\documentclass[sigconf]{acmart}`. Confirm page limit and blind/non-blind on the portal in Phase 0.

---

## 1. Thesis

Video transformers tokenize a clip as a fixed grid of `T × N` spatiotemporal patches. Those tokens have no
persistent identity: patch `(i, j)` at frame `t` and at frame `t+1` are not the same entity, objects move
across the grid, and adjacent frames are near-duplicates. Attention therefore pays `O((TN)²)` to compute
affinities between tokens carrying no stable referent — unlike text, where a token at position 3 and one at
position 40 denote the same lexical item, which is precisely what makes attention between them meaningful.

**EPT replaces the grid with tokens that persist.** Detect and track people across the clip; emit one token
per `(entity, temporal segment)`. A token then denotes *this person during this interval*, and attention
splits into two interpretable operations: **temporal** attention within an entity's track (how one person's
state evolves) and **social** attention across entities at matched time (who co-varies with whom). Token
count falls from `T × N` (16 × 196 = 3136) to `E × S` (8 × 8 = 64).

---

## 2. Pre-registration

Committed to `docs/DECISION_RULES.md` **before** any test-split evaluation. Config selection uses validation
only; the test split is touched once per locked config.

**Primary endpoint.** Test macro-F1 on OUC-CGE, mean over 3 seeds `{42, 1337, 2024}`, existing 771-clip
video-only protocol.

**H1 (persistence).** `A1 (EPT) − A2 (identity-shuffled EPT) ≥ +0.02` macro-F1.
**H2 (entity structure).** `A1 (EPT) − A0 (grid tokens, matched budget) ≥ +0.02` macro-F1.

| Outcome | Paper branch |
|---|---|
| H1 supported | **(a)** *Entity-persistent tokenization* — persistence is the mechanism. Headline = A1 vs A2. |
| H1 null (`\|A1−A2\| < 0.02`), H2 supported | **(b)** *Boundary paper* — persistence is **not** what helps; entity-localized cropping and token budget are. Headline = A1/A2 vs A0. |
| H1 refuted (`A1 < A2 − 0.02`) | **(c)** Report as refuted; paper pivots to the efficiency Pareto result (§6), negative finding as the contribution. |

All three branches are publishable. The paper does not depend on the hypothesis being true.

**Also pre-registered:** no learned per-slot identity embedding (it would leak slot ordering). Slot assignment
order is randomized per sample in **every** condition including A2, so A1-vs-A2 isolates *temporal identity
consistency* and nothing else.

---

## 3. Architecture

### 3.1 Tokenization
```
clip → frame sampling (T=32 @ ~3fps for OUC-CGE)
     → face detection per frame (SCRFD / buffalo_l; fallback: YOLO person detection)
     → ByteTrack ID association across frames
     → keep top-E tracks by (mean confidence × frame coverage), E_max = 8
     → crop each track per frame → 224²
     → frozen DINOv2 → per-frame embedding
     → mean-pool within each of S=8 segments
     → token grid x[e, s] ∈ R^d,  e ≤ 8, s = 8   →  ≤ 64 tokens
```
Absent entries (entity not present in a segment) get a learned `[ABSENT]` embedding plus an attention mask.

### 3.2 EPT-Former
- `d = 384`, `L = 4` blocks, 6 heads, GELU, pre-norm.
- Each block: **temporal attention** over `s` at fixed `e` → **social attention** over `e` at fixed `s` → FFN
  (mlp_ratio 4).
- Additive embeddings: sinusoidal segment position + `[ABSENT]` flag. **No** per-slot identity embedding.
- Readout: learned `[CLS]` cross-attending over valid `(e,s)` tokens → LayerNorm → linear → 3 classes.
- Cost: `O(E·S² + S·E²)` ≈ 1.0k pairwise terms per block, vs ~9.8M for full space-time attention at T=16.

### 3.3 Frozen backbone
`facebook/dinov2-with-registers-base` (ViT-B/14). Representation = CLS ⊕ mean of patch tokens. The **same**
frozen features feed **every** condition including the grid baseline — matched-feature discipline, so any
difference between conditions is attributable to tokenization and attention topology, not encoder capacity.
Features extracted once and cached as `.npy`; the whole ablation matrix then trains in minutes per run.

---

## 4. Data

| Dataset | Role | Notes |
|---|---|---|
| **OUC-CGE** | **Primary** — multi-person classroom, group engagement, 3 classes (2396/1653/2115) | CC-BY 4.0. Fixed camera, no shot cuts, many simultaneous entities — the conditions the claim needs. Video-only, 771-clip protocol. Exclude `low/view2572.mp4` (missing) and `low/view2531.mp4` (moov atom not found). |
| **DAiSEE** | Control — single subject, `E = 1` | Shows EPT degenerates gracefully to a temporal-only model and that any gain is specifically multi-entity. Cheap; reuses the same cache pipeline. |
| **MELD** | **Conditional** generalization set — multi-party, non-classroom | Only if the Phase 5 gate passes (§7). **Requires raw video on disk** — SocialArcNet-extracted features are insufficient, entity crops need original frames. Method variant: MELD is edited television, so shot cuts shatter IoU tracking — use **face-embedding clustering within the clip** to recover identity instead of ByteTrack. Primary endpoint on MELD is **3-class sentiment**, not 7-class emotion, for signal reasons. |
| EngageNet | Not used | Single-subject; adds nothing over DAiSEE for this question. |

**Overlap management (OUC-CGE ∩ NCAA paper).** Different research question (tokenization and compute, not
benchmark accuracy), different measurements, different tables. Cite the NCAA paper as under review. Requires
explicit sign-off from Atadjanov and Abdulali in Phase 0 before any runs.

Audio excluded throughout (video-only), consistent with the 771-clip protocol.

---

## 5. Experiment matrix

Shared: frozen DINOv2 features, cached; identical optimizer recipe (locked end of Phase 3); 3 seeds.

| ID | Condition | What it isolates |
|---|---|---|
| **A0** | Grid-patch tokens, matched 64-token budget | Entity structure vs none |
| **A1** | **EPT (full)** | Primary condition |
| **A2** | EPT, entity IDs shuffled independently per segment | **Persistence itself** — the key control |
| **A3** | EPT, temporal attention only | Contribution of social attention |
| **A4** | EPT, social attention only | Contribution of temporal attention |
| **A5** | Mean-pool entity tokens → MLP | Is attention needed at all? |
| **A6** | **Reference row, NOT feature-matched** — verified TimeSformer/VideoMAE numbers from the BPAVTforSGER reruns on this exact 771-clip video-only OUC-CGE protocol | Accuracy-ceiling *context*, not part of the matched-feature ablation |

Secondary sweep from the same cache: `E ∈ {1,2,4,8,16}` × `S ∈ {2,4,8,16}` → token-budget Pareto
(`E=16` per the 2026-08-14 `DECISION_RULES.md` amendment; primary condition A1 stays locked at `E=8`).

**A6 redefinition (2026-08-14, Phase 3).** A6 was originally scoped as a "frozen-feature probe"
implying it would share the frozen-DINOv2 cache like every other condition — but no such probe was
ever built for TimeSformer/VideoMAE, and building one now would mean a *different* frozen backbone
(TimeSformer/VideoMAE's own encoder, not DINOv2) probed the same way, which is not a meaningful
comparison and was never actually planned. **A6 is redefined as a reference row**: verified numbers
from the BPAVTforSGER benchmark suite's reruns on the identical 771-clip video-only OUC-CGE test
protocol, marked in every results table as **not feature-matched** (non-negotiable #3 is explicitly
waived for this row only — it exists for accuracy-ceiling context, not as an ablation arm).

| Model | Regime | Macro-F1 | Seeds | Source |
|---|---|---:|---|---|
| TimeSformer | Fine-tuned (full backbone) | 0.9917 | 1 (no ± std available) | `BPAVTforSGER/outputs/OUC-CGE/timesformer_finetune/` |
| TimeSformer | Video-only linear probe (frozen backbone) | 0.2648 | 1 (no ± std available) | `BPAVTforSGER/outputs/OUC-CGE/timesformer_linprobe/` |
| VideoMAE | Fine-tuned (full backbone) | 0.9875 | 1 (no ± std available) | `BPAVTforSGER/outputs/OUC-CGE/videomae_experiment/` |
| VideoMAE | Video-only linear probe (frozen backbone) | **not available** — no linear-probe rerun exists for VideoMAE in BPAVTforSGER; report the gap, do not fabricate a number | — | — |

The fine-tuned numbers (TimeSformer 0.992, VideoMAE 0.988) show the task is close to saturated
*given full end-to-end fine-tuning* — useful ceiling context. The frozen linear-probe number
(TimeSformer 0.265, barely above the 0.190 majority-class floor and below the ~0.33 stratified-random
floor established in Phase 2) is the more relevant comparison for A0–A5: it shows a frozen backbone
probed the crude way (no tokenization structure at all, just a linear head on pooled features) does
**not** get you the fine-tuned ceiling — which is exactly the regime A0–A5 operate in, and is why the
frozen-probe number belongs in the table explicitly rather than only the flattering fine-tuned one.

---

## 6. Efficiency protocol

Reported **end-to-end**. Entity tokenization does not remove computation, it relocates it into the detector;
hiding that is the fastest way to lose a reviewer.

- Token count and GFLOPs of the attention stack (ptflops / fvcore).
- **Plus** detection + tracking cost per clip, measured separately, reported as its own line item.
- Latency: RTX 5090 (batch 1, fp16) and CPU-only at `threads ∈ {1, 8, 60}` — the CPU figures stand in for
  edge-server deployment and are what the IoT-track audience cares about.
- Peak memory; accuracy-vs-GFLOPs Pareto curve as Figure 3.

---

## 7. Phase plan

Phase gates: no phase starts until the previous gate criterion is written to the log.

| Phase | Dates | Work | Gate |
|---|---|---|---|
| **0** | Aug 14–15 | HF connectivity test; disk check; repo init; commit `DECISION_RULES.md`; portal check (page limit, blind/non-blind); coauthor sign-off | Decision rules committed **before** any run; DINOv2 weights resolved locally |
| **1** | Aug 16–18 | Detection + ByteTrack on OUC-CGE; tracking QA report | ≥80% of clips yield ≥2 tracks with ≥50% frame coverage. **Else → §8 fallback** |
| **2** | Aug 19–20 | DINOv2 feature extraction & caching (OUC-CGE + DAiSEE) | Cache complete, checksummed, spot-verified on 5 clips; rclone backup |
| **3** | Aug 21–23 | EPT-Former implementation; recipe search on **val only**; lock | `LOCKED_RECIPE.md` committed; no further tuning |
| **4** | Aug 24–26 | Full matrix A0–A6 × 3 seeds; token-budget sweep | All runs logged; **decision rule fires once** |
| **5** | Aug 24–27 | *Conditional:* MELD extension | **Gate: only if Phase 4 is on schedule as of Aug 24 AND MELD raw video confirmed on disk.** Otherwise skipped without penalty. |
| **6** | Aug 27–28 | Efficiency measurements; figures; qualitative attention maps | Figures 1–4 final |
| **7** | Aug 21–30 | Writing (runs in parallel from Phase 3) | Full draft circulated Aug 29 |
| **8** | Aug 31 | Coauthor review; buffer day | Sign-off from all authors |
| **9** | Sep 1 | Formatting check, references, submission via Sparcly | Submitted |

Sections 1–3 (intro, related work, method) do not depend on results — draft them from Aug 21.

---

## 8. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| **Tracking fails** on small / occluded classroom faces | Switch face → full-person detection. If still poor, fall back to *region-persistent tokens* (fixed seat-region grid + per-region temporal pooling). The claim survives as "region-persistent" rather than "entity-persistent", with the tracking failure reported as a finding. |
| HF unreachable from the server | Download `facebook/dinov2-with-registers-base` on the laptop, `scp` the folder, point `from_pretrained` at the local path. |
| Lab server / IT instability | Cache features to disk in Phase 2 and rclone-backup immediately. After Phase 2 the whole matrix runs on cached `.npy` — a dead GPU costs hours, not the paper. |
| Disk pressure (IEMOCAP already resident) | Check free space in Phase 0. Entity-crop caches are small (~1–2 GB); intermediate crops should be streamed, not stored. |
| MELD is features-only on disk | Phase 5 is skipped. The paper stands on OUC-CGE + DAiSEE. |
| `E = 1` datasets show no effect | Expected and pre-registered — that is what makes DAiSEE a control rather than a failure. |
| Reviewers read it as "just pooling over face crops" | A2 (identity shuffle) is the answer, and it is the headline comparison for exactly this reason. |

---

## 9. Locked technical choices

- Backbone: `facebook/dinov2-with-registers-base`, frozen, fp16 inference.
- Detector: InsightFace SCRFD (`buffalo_l`); fallback YOLO person class.
- Tracker: `supervision` ByteTrack (no weights → no network dependency).
- Seeds: `{42, 1337, 2024}` — consistent with the SocialArcNet recipe.
- `E_max = 8`, `S = 8`, `T = 32`, `d = 384`, `L = 4`.
- Reporting: mean ± std across three seeds, every number.
- **Software stack amendment (Phase 0, 2026-08-14):** `PyTorch 2.4` (as specified in `CLAUDE.md` §Stack)
  predates Blackwell (`sm_120`) kernel support and fails on this lab's RTX 5090 ("no kernel image is
  available for execution on the device"). Substituted `torch==2.7.0+cu128` — the same build the
  BPAVTforSGER benchmark suite already trains successfully with on this machine. Verified via
  `torch.cuda.get_device_capability() == (12, 0)` and a real fp16 matmul executed on-device.
  `transformers>=4.45` and the rest of the stack are unaffected.

---

## 10. Venue and framing

**Track: Internet of Things (IoT).** Its published topic list covers *data collection, deep learning and
prediction methods*, *real-time decision-making and information extraction*, *information architecture design
on field device, edge device and servers in cloud*, and *deployment case studies* — all of which this paper
hits directly. The General Track's only relevant entry is "Artificial intelligence."

**Framing:** efficient on-device video analytics for instrumented learning spaces. The CPU latency numbers
(§6) are what make the framing credible rather than decorative — do not cut them under time pressure.

**Working titles**
1. *Tokens That Persist: Entity-Centric Video Tokenization for Group Engagement Analysis at the Edge*
2. *Does Identity Persistence Matter? Entity-Persistent Tokenization for Efficient Classroom Video Understanding*

Title 2 if the decision rule fires branch (b) or (c).
