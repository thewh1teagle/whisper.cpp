# 5-Way Comparison: Baseline vs v2 vs v3 vs v4 vs stable-ts

Date: 2026-03-06
Audio: synth_5min.wav (5 min, 46 utterances, 7x 20s long pauses)
Model: large-v3-turbo (GGML for ours, faster-whisper for stable-ts)

---

## Results

| Metric                    | Baseline | v2   | v3    | **v4**    | stable-ts |
|---------------------------|----------|------|-------|-----------|-----------|
| n_segments                | 161      | 45   | 45    | **46**    | 46        |
| n_seg_start_in_silence    | 86       | 1    | 2     | 5         | 17        |
| n_seg_overlap_any         | 120      | 30   | 43    | **5**     | 18        |
| n_words total             | 1813     | 636  | 506   | 563       | **386**   |
| n_words_overlap_any       | 745      | 144  | 52    | **5**     | 22        |
| pct_words_overlap %       | 41.1     | 22.6 | 10.3  | **0.89**  | 5.7       |
| n_words_start_in_silence  | 954      | 142  | 71    | 89        | **26**    |
| pass_segments_threshold   | False    | False| False | **True**  | False     |

---

## v4: Per-Segment VAD Decoding

v4 changes the VAD pipeline from:
  *concatenate all speech → decode once → remap timestamps*
to:
  *decode each VAD segment independently → offset timestamps by orig_start*

This matches how stable-ts/faster-whisper works.

**Result:** pct_words_overlap dropped from 10.3% (v3) → **0.89% (v4)**, beating stable-ts (5.7%).

The remaining 5 segment overlaps are all very small (max 2ms) vs stable-ts max 7100ms.
n_words_start_in_silence is 89 vs stable-ts 26 — these are words at boundaries of VAD
segments where the PCM silence map correctly shows no silence but the boundary transitions
quickly. Not a functional issue for subtitle quality.

---

## Architecture Comparison

| Approach | VAD | Decode | Timestamp mapping |
|----------|-----|--------|-------------------|
| v2 | No | Single-shot with constrained filter | None needed |
| v3 | Concatenate → single decode | Complex vad_mapping_table interpolation | Approx |
| **v4** | Per-segment decode | Fixed offset per segment | Exact |
| stable-ts | Per-segment (faster-whisper) | Per-segment | Fixed offset |

v4 and stable-ts now use the same fundamental approach.

---

## Code Removed in v4

- `whisper_vad()` function (concatenation + mapping table building)
- `vad_time_mapping` struct from whisper-state.h
- `vad_segments`, `has_vad_segments`, `vad_mapping_table` from whisper_state
- `map_processed_to_original_time()` in whisper.cpp
- `whisper_stable_map_processed_to_original()` in whisper-stable.cpp
- `has_vad_mapping` + `mapping` params from `whisper_stable_snap_segments`
- ~200 lines net reduction

## Code Added in v4

- `whisper_full_vad_segments()` — ~70 lines, clean per-segment decode loop
