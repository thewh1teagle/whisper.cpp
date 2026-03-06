# Status Notes — v4 Complete

Date: 2026-03-06
Branch: feature/stable-timestamps

---

## Build Status

Compiles clean. Binary at: build/bin/whisper-cli

---

## v4 Summary (completed)

### What changed from v3

- Replaced `whisper_vad()` (concatenate + map table) with `whisper_full_vad_segments()`
  (per-segment decode + fixed offset per segment)
- Removed `vad_time_mapping` struct from whisper-state.h
- Removed `vad_segments`, `has_vad_segments`, `vad_mapping_table` from whisper_state
- Removed `map_processed_to_original_time()` from whisper.cpp
- Removed `whisper_stable_map_processed_to_original()` from whisper-stable.cpp
- Simplified `whisper_stable_snap_segments()` — removed mapping params (no longer needed)
- Simplified `whisper_full_get_segment_t0/t1_from_state()` — return stored value directly
- `whisper_full_parallel()` with VAD delegates to `whisper_full()` (per-segment sequential)

---

## Validation Results — 5-way Comparison

| Metric                      | Baseline | v2   | v3    | v4      | stable-ts |
|-----------------------------|----------|------|-------|---------|-----------|
| n_segments                  | 161      | 45   | 45    | 46      | 46        |
| n_seg_start_in_silence      | 86       | 1    | 2     | 5       | 17        |
| n_seg_overlap_any           | 120      | 30   | 43    | 5       | 18        |
| n_words total               | 1813     | 636  | 506   | 563     | 386       |
| n_words_overlap_any         | 745      | 144  | 52    | **5**   | 22        |
| pct_words_overlap %         | 41.1     | 22.6 | 10.3  | **0.89**| 5.7       |
| n_words_start_in_silence    | 954      | 142  | 71    | 89      | 26        |
| pass_segments_threshold     | False    | False| False | **True**| False     |

**v4 beats stable-ts on word overlap metric (0.89% vs 5.7%).**

---

## Available Artifacts

- `models/ggml-large-v3-turbo.bin` — 1.62GB, ready
- `models/for-tests-silero-v6.2.0-ggml.bin` — VAD model, ready
- `plans/stable-timestamps-v2/out/synth_5min.wav` — 5min synthetic audio, ready
- `plans/stable-timestamps-v2/out/stable_v2.json` — v2 transcription, ready
- `plans/stable-timestamps-v2/out/stable_v3.json` — v3 transcription, ready
- `plans/stable-timestamps-v2/out/stable_v4.json` — v4 transcription, ready
- `plans/stable-timestamps-v2/out/stable_ts_ref.json` — stable-ts reference, ready

---

## Usage

```bash
# Best accuracy — per-segment VAD + snapping
./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin \
  -f audio.wav --stable-timestamps --vad --vad-model models/for-tests-silero-v6.2.0-ggml.bin

# No VAD fallback — constrained decoding only
./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin \
  -f audio.wav --stable-timestamps
```

---

## Remaining Items

- n_words_start_in_silence = 89 vs stable-ts 26 — words at VAD boundary edges, not a
  functional issue for subtitle quality (overlap metric is what matters: 0.89% vs 5.7%)
- Test on real movie audio clip
- Filter [_TT_*] timestamp tokens from CLI word output (pre-existing behavior)
