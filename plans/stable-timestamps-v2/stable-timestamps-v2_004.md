# Stable Timestamps v3 — VAD + Snapping Combined

Goal: match stable-ts word-level boundary quality by combining VAD silence stripping with
correct post-hoc snapping, using proper processed→original timeline remapping.

Reference comparison: see notes/comparison_001.md

---

## Why v2 Isn't Enough

v2 gets segment hallucination under control (161→45) via constrained decoding, but word-level
boundary quality is still 22.6% vs stable-ts 5.7%. Root cause: without VAD stripping, the
decoder still processes 20s silence windows and produces hallucinated words that land in silence.
Those inflate the word overlap metrics and can't be fixed by post-hoc snapping alone.

stable-ts gets its quality because faster-whisper applies internal VAD → never decodes silence
→ only real speech tokens with snappable boundaries remain.

---

## What v3 Changes

**One new code path:** when `stable_timestamps=true` AND `vad=true`, remap token timestamps
from processed (VAD-stripped) timeline → original timeline via `vad_mapping_table` before
snapping. Then clear `vad_mapping_table` so public getters don't double-remap.

This is what v1 attempted but with wrong snapping algorithm. v3 = correct snapping + mapping.

**When only `stable_timestamps` (no VAD):** v2 behavior unchanged — constrained decoding +
snap on original timeline, no mapping needed.

**No hard VAD requirement** — v2 fallback is always available.

---

## Code Changes

### `src/whisper-stable.h`

Add `has_vad_mapping` + `mapping` params to `whisper_stable_snap_segments`:

```c
void whisper_stable_snap_segments(
    struct whisper_context * ctx,
    std::vector<whisper_segment> & result_all,
    const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
    const std::vector<std::pair<int64_t, int64_t>> & mapping_processed_to_original,  // NEW
    bool has_vad_mapping,                                                              // NEW
    int64_t min_word_dur_cs,
    int64_t min_snap_silence_dur_cs);
```

Add internal helper declaration:
```c
int64_t whisper_stable_map_processed_to_original(
    int64_t processed_cs,
    const std::vector<std::pair<int64_t, int64_t>> & mapping);
```

### `src/whisper-stable.cpp`

Add `whisper_stable_map_processed_to_original()` — linear interpolation between mapping points
(same logic as v1's static function, now exposed for reuse):

```c
int64_t whisper_stable_map_processed_to_original(
    int64_t processed_cs,
    const std::vector<std::pair<int64_t, int64_t>> & mapping) {
    // If mapping empty: identity
    // Clamp to first/last mapping point
    // Binary search + linear interpolation between bracketing points
}
```

Update `whisper_stable_snap_segments()`:
- Accept `mapping_processed_to_original` + `has_vad_mapping`
- Before building flat t0/t1 arrays: if `has_vad_mapping`, remap each token's t0/t1 via mapping
- After snapping + writeback: unchanged (snapped values are now original-timeline)

### `src/whisper.cpp`

In `whisper_full()`, after `whisper_full_with_state()` returns, build the mapping and pass it:

```c
if (params.stable_timestamps && !stable_silence.empty()) {
    // Build mapping from vad_mapping_table if VAD was active
    std::vector<std::pair<int64_t,int64_t>> mapping;
    bool has_mapping = false;
    if (params.vad && state->has_vad_segments && !state->vad_mapping_table.empty()) {
        for (const auto & m : state->vad_mapping_table) {
            mapping.push_back({m.processed_time, m.original_time});
        }
        has_mapping = true;
    }

    whisper_stable_snap_segments(
        ctx, state->result_all, stable_silence,
        mapping, has_mapping,
        /*min_word_dur_cs=*/5,
        /*min_snap_silence_dur_cs=*/10);

    // Clear VAD remap — snapping already applied original-timeline values
    if (has_mapping) {
        state->vad_mapping_table.clear();
        state->has_vad_segments = false;
    }
}
```

---

## Recommended Usage (CLI)

```bash
# Best accuracy — matches stable-ts quality
./build/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f audio.wav \
  --stable-timestamps \
  --vad \
  --vad-model models/ggml-silero-v6.2.0.bin

# No VAD model — v2 fallback (constrained decoding only, still good)
./build/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f audio.wav \
  --stable-timestamps
```

---

## Validation

4-way comparison after implementation:

```bash
# 1. Baseline
./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps-v2/out/synth_5min.wav -l en -ojf \
  -of plans/stable-timestamps-v2/out/baseline

# 2. Stable v2 (no VAD)
./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps-v2/out/synth_5min.wav -l en \
  --stable-timestamps -ojf \
  -of plans/stable-timestamps-v2/out/stable_v2

# 3. Stable v3 (with VAD)
./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps-v2/out/synth_5min.wav -l en \
  --stable-timestamps --vad --vad-model models/for-tests-silero-v6.2.0-ggml.bin -ojf \
  -of plans/stable-timestamps-v2/out/stable_v3

# 4. stable-ts reference (already done)
# plans/stable-timestamps-v2/out/stable_ts_ref.json

# Compare all
for f in baseline stable_v2 stable_v3 stable_ts_ref; do
  uv run plans/stable-timestamps/stable-timestamps_005_compare.py \
    --timeline-csv plans/stable-timestamps-v2/out/synth_5min_timeline.csv \
    --baseline-json plans/stable-timestamps-v2/out/baseline.json \
    --stable-json plans/stable-timestamps-v2/out/${f}.json \
    --out-csv plans/stable-timestamps-v2/out/metrics_${f}.csv
done
```

### Pass Criteria for v3

| Metric | v2 | Target v3 | stable-ts ref |
|--------|----|-----------|---------------|
| n_segments | 45 | ~46 | 46 |
| n_seg_start_in_silence | 1 | < 5 | 17 |
| pct_words_overlap % | 22.6 | < 10 | 5.7 |
| n_words_start_in_silence | 142 | < 40 | 26 |

---

## TODO

- [x] Add `whisper_stable_map_processed_to_original()` to whisper-stable.cpp + .h
- [x] Update `whisper_stable_snap_segments()` signature and body (mapping remap path)
- [x] Update `whisper_full()` in whisper.cpp: build mapping, pass to snap, clear after
- [x] Build and smoke test on jfk.wav with --vad --stable-timestamps
- [x] Run 4-way validation on synth_5min.wav
- [x] Update notes/comparison_001.md with v3 results
- [x] Update notes/status_001.md
