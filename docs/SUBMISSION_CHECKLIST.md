# Submission Checklist — ICFNDS 2026

Status as of 2026-08-16, checked against the current state of `paper/` and
`docs/` rather than assumed. Each item below was verified by reading the
actual file, not recalled from memory; the evidence is cited so it can be
re-checked after further edits. None of these are done unless marked done.

## 1. CCS concepts block

**Not verified. Blocked on browser/tool access, unresolved.**

`paper/main.tex`'s `\begin{CCSXML}` block (lines 74–102) has five
`concept_id` values. Only one — "Computing methodologies~Activity
recognition and understanding" (`10010147.10010178.10010224.10010225.10010228`)
— was cross-checked this session, against a real paper's published CCSXML
source, and corrected once in the process (an earlier guess was missing the
final `.10010228` segment). The other four (`General and reference~Evaluation`,
`Computing methodologies~Neural networks`, `General and reference~Measurement`,
`Computer systems organization~Embedded systems`) are unverified guesses.
`dl.acm.org/ccs.cfm` returned 403 and `dlccsdev.acm.org` did not resolve on
every attempt made this session (see the `%% TODO before submission` comment
directly above the CCSXML block in `main.tex`, lines 65–73, which states this
explicitly).

**Action:** regenerate the full block from the live ACM CCS tool from a
network/browser this account can reach, and diff it against what's currently
in the file before treating any of the five IDs as final.

## 2. Author list and order

**Not confirmed. Single placeholder author, explicit TODO in source.**

`paper/main.tex` line 36: `%% TODO: confirm the author list and order with
all coauthors before submission.` Currently lists one author (Abdulaziz
Xo'jamqulov, Tashkent University of Information Technologies) with an empty
`\email{}` field. No coauthors are listed.

**Action:** confirm final author list, order, affiliations, and email
addresses with every coauthor; fill in `\email{}`.

## 3. Ethics statement / OUC-CGE consent attribution

**Not present. This is a gap, not a formatting detail — flagging directly.**

Checked: no occurrence of "ethic", "consent", or "IRB" anywhere in
`paper/main.tex` or `paper/sections/*.tex`. There is no ethics statement or
ethical-considerations section in the current draft.

Also checked the specific claim that this attribution lives in "the Figure 2
caption": Figure 2 in the current draft (`\label{fig:seed-margin}`,
`results.tex` lines 77–84) is "Per-seed H1/H2 margins (MELD), item-paired
bootstrap vs. seed-paired $t$ interval" — a statistics figure with no
connection to OUC-CGE or consent. There is no figure anywhere in the paper
whose caption currently carries a consent/data-provenance attribution for
OUC-CGE. OUC-CGE is discussed only in prose, in §3.1 (`dataset_audit.tex`)
as the rejected-dataset case study, and in `refs.bib` as `lu2025ouccge`.

**Action:** two separate things need to happen, and neither exists yet:
(a) add an ethics statement (ACM sigconf papers typically carry this as an
unnumbered section before the bibliography, or via `\acks`/a dedicated
`\section*{Ethical Considerations}`) covering data-consent status for
MELD, DAiSEE, and OUC-CGE as used; (b) decide which figure or table is
meant to carry the OUC-CGE consent attribution — Table 1
(`\label{tab:ouccge-dup}` in `dataset_audit.tex`, the near-duplicate cosine-
similarity table) is the only OUC-CGE-specific float in the paper and is
the more natural home for it than Figure 2, but this needs a decision, not
an assumption on my part. Confirm with whoever holds OUC-CGE's original
consent/licensing terms before writing the attribution language.

## 4. `refs.bib` completeness

**Not complete. Concrete gap found: DINOv2 is cited by name in the text with no bibliography entry.**

Checked every `\cite{}` call against `paper/refs.bib`: all resolve (0
undefined-citation warnings in the final `bibtex`/`pdflatex` build), so
nothing currently *breaks*. But `refs.bib`'s own header comment (lines 1–9)
lists five references as "still needed, not yet added": DINOv2 (Oquab et
al.), VideoMAE (Tong et al., the A6 reference), ByteTrack (Zhang et al.),
InsightFace/SCRFD (Guo et al.), and the BPAVTforSGER paper (cited as under
review).

Of those five, **DINOv2 is a real, present gap**: the paper body names it
twice in prose — `results.tex` line 162 ("The frozen DINOv2 encoder...")
and `discussion.tex` line 39 ("the same frozen DINOv2 features") — with no
`\cite{}` attached either time. A frozen backbone this central to every
condition in the paper being named without a citation is worth fixing before
submission, not just before camera-ready. SCRFD/ByteTrack are named in
`CLAUDE.md`'s stack description but not yet in the paper prose itself, so
they're lower-priority unless the Method section is extended to name the
detector/tracker explicitly. No placeholder or malformed entries were found
among the 12 entries currently in the file — every existing entry has real
author/title/venue/year fields.

**Action:** add an Oquab et al. DINOv2 citation and attach it to both
mentions; decide whether SCRFD/ByteTrack need citing given the current
prose doesn't name them; resolve the BPAVTforSGER "under review" citation's
status before submission.

## 5. Repository URL

**Not included anywhere in the paper.**

Checked `paper/main.tex` and all of `paper/sections/*.tex` for any
GitHub/repo/code-availability mention — none found. The `\begin{acks}`
block (main.tex lines 140–142) is currently empty by design ("Omit entirely
in anonymous mode"), which is also where a code-availability statement would
typically go for a non-anonymous camera-ready version.

**Action:** decide whether the repo (`github.com/multi-modal-rtm/ept`, per
this project's earlier git-remote setup) is intended to be public and cited
in the paper, and if so add a code-availability statement — but only once
review-mode (anonymous vs. non-anonymous) is confirmed per item 7, since an
anonymous submission cannot point at an identifying repo URL.

## 6. acmart metadata TODOs (ISBN, DOI, conference string)

**Already flagged in source as camera-ready items — confirming that flagging is in place, not resolving it.**

All three are marked in `paper/main.tex`:
- Line 12: `%% TODO: replace with the exact strings from the ICFNDS
  camera-ready instructions. Do not guess the edition number of the
  conference.`
- Line 19: `\acmISBN{978-x-xxxx-xxxx-x/26/11}   % TODO`
- Line 20: `\acmDOI{10.1145/nnnnnnn.nnnnnnn}    % TODO`

These are correctly placeholder values (ACM's standard unfilled format, not
guessed real numbers) and are correctly marked as pending. No action needed
before initial submission — ACM fills ISBN/DOI at camera-ready time, not
submission time, for most sigconf venues, but **this project has not itself
confirmed that's true for ICFNDS specifically** (see item 7).

## 7. Page limit confirmed against the portal

**Not confirmed. Flagged as a to-do since Phase 0, still open.**

`docs/PLAN.md` line 6: "Confirm page limit and blind/non-blind on the portal
in Phase 0." Line 201 repeats this as a Phase 0 deliverable ("portal check
(page limit, blind/non-blind); coauthor sign-off"). Checked `logs/GATES.md`
for any record that this was actually done — found no entry recording a
page-limit or blind-mode confirmation from the ICFNDS submission portal at
any point in this project's history. `paper/main.tex` line 2–3 still carries
the caveat comment: "For blind review (CHECK THE PORTAL FIRST — ICFNDS has
historically been non-blind)" — worded as an open warning, not a resolved
fact.

The current draft is **9 pages** (body + references, single build; see the
compile report delivered alongside this checklist) under `\documentclass[sigconf]{acmart}`,
non-anonymous mode. Whether 9 pages is within ICFNDS's actual limit is
unverified.

**Action:** this is the one item on this list that requires an external
source (the ICFNDS submission portal or CFP page) rather than a repo-internal
check. Someone needs to open the portal, confirm the page limit and blind/
non-blind requirement, and record it here or in `docs/PLAN.md` before
submission — it hasn't happened yet at any point in this project.

---

**Summary:** 0 of 7 items are done. Item 6 is correctly *staged* (real
placeholders, properly marked) but not resolvable pre-camera-ready by
definition. Items 1, 2, 3, 4, 5, 7 all require action — 3 and 7 in
particular need a decision or lookup from outside this repo (OUC-CGE's
consent/licensing status, and the ICFNDS portal itself) before they can be
closed.
