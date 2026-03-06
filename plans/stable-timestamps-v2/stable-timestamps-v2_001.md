# Stable Timestamps v2 — Clean Rewrite Plan

Goal: subtitle-quality word timestamps matching stable-ts accuracy. No backward compatibility with v1.

---

## What's Wrong in v1

### 1. Snapping algorithm is fundamentally broken
`whisper_stable_snap_timestamps` finds the speech region with maximum overlap with the word
and replaces the entire word's t0/t1 with that region's boundaries. This destroys word spans.
stable-ts is surgical: only move the specific boundary that falls inside silence. Keep everything else.

### 2. Hard VAD requirement creates a catch-22 for Phase 3
v1 forces `vad=true` when `stable_timestamps=true`. VAD strips silence before decoding. On the
processed timeline there is no silence. The timestamp silence mask for constrained decoding is
nearly all-zeros — Phase 3 does nothing.

### 3. Silence map granularity is too coarse
VAD probs are 32ms/frame. Inter-word pauses are 20–80ms. We miss them. stable-ts uses raw PCM
energy at 20ms/bin (320 samples, same as Whisper's audio token resolution). Much more sensitive.

### 4. Plan was philosophically confused
stable-ts never strips silence — it runs Whisper on full audio and corrects timestamps post-hoc.
Requiring VAD stripping fights against the purpose of snapping.

---

## What We Throw Out

- [ ] `whisper_stable_snap_timestamps` — entire body, rewrite from scratch
- [ ] `whisper_stable_build_silence_map` (VAD-prob version) — replace with PCM-energy version
- [ ] `whisper_stable_prepare` / `whisper_stable_prepare_from_ctx` / `whisper_stable_finalize` — delete
- [ ] `whisper_stable_runtime` struct — delete
- [ ] `whisper_stable_build_timestamp_silence_mask` — delete (timeline mapping complexity gone)
- [ ] `whisper_stable_copy_vad_mapping` — delete
- [ ] `whisper_stable_map_processed_to_original_time` (static) — delete
- [ ] `whisper_stable_setup_filter` — simplify (no mapping table)
- [ ] The VAD forcing in `whisper_full()` (`params.vad = true`) — remove
- [ ] The `vad_mapping_table.clear()` trick in finalize — remove

## What We Keep

- [ ] `whisper_stable_get_gap_tokens` — correct, keep as-is
- [ ] `whisper_stable_select_heads` — correct, keep as-is
- [ ] `whisper_stable_silence_overlap_len` — utility, keep
- [ ] `whisper_stable_logits_filter_callback` — keep, simplify (no mapping)
- [ ] DTW wiring in `whisper.cpp` (gap tokens + head selection) — correct, keep

---

## New Design

### Silence map: raw PCM energy (not VAD probs)

New function `whisper_stable_build_silence_map_from_pcm()` mirrors stable-ts `wav2mask` exactly:

```
1. abs(pcm) for all samples
2. normalize by 99.9th percentile (top 0.1% value)
3. interpolate down to 1 value per audio token position (320 samples = 20ms)
4. avg-pool with kernel=5, reflection padding (smoothing)
5. quantize: (value * 20).round() → anything rounding to 0 = silent
6. convert to boolean mask
7. merge adjacent silent frames into regions, filter out regions < min_silence_dur
8. return vector of (start_cs, end_cs) pairs in centiseconds
```

This gives 20ms resolution, catches inter-word pauses, no VAD model required.

Signature:
```c
std::vector<std::pair<int64_t, int64_t>> whisper_stable_build_silence_map_from_pcm(
    const float * pcm,
    int           n_samples,
    int           sample_rate,
    int64_t       min_silence_dur_cs);  // e.g. 10 = 0.1s
```

### No VAD stripping required

`stable_timestamps = true` forces only:
- `token_timestamps = true`
- `max_initial_ts = 0.0f`

VAD stripping stays independent. User can combine or not. No error if vad_model_path is missing.

### Correct snapping algorithm

Rewrite `whisper_stable_snap_timestamps` to match stable-ts exactly.

For each word [t0, t1] in order:

```
1. START IN SILENCE
   Find silence region [s, e] where s <= t0 < e and e <= t1
   → t0 = e  (move start forward to speech edge, keep end)

2. END IN SILENCE
   Find silence region [s, e] where t0 <= s < t1 <= e
   → t1 = s  (move end backward to speech edge, keep start)

3. SILENCE FULLY INSIDE WORD
   Find silence region [s, e] where t0 < s && e < t1
   → compute left_ratio  = (s - t0) / (e - s)  [how much speech before silence]
   → compute right_ratio = (t1 - e) / (e - s)  [how much speech after silence]
   → if first word in segment: prefer snapping start forward (t0 = e)
   → if last word in segment:  prefer snapping end backward (t1 = s)
   → otherwise: snap the side with higher ratio (more speech = more important to keep)

4. CLAMP
   If new_t1 - new_t0 < min_word_dur_cs:
   → restore from the anchored side (keep t0 if start was moved, keep t1 if end was moved)
   → ensure at least min_word_dur_cs duration

5. After all words done: update segment t0 = first_word.t0, segment t1 = last_word.t1
```

Multiple silence regions per word: process them in time order, one at a time.

### Constrained decoding — now actually works

Since we no longer VAD-strip audio, processed timeline IS original timeline. No mapping needed.

Silence mask is built from PCM energy on the full audio. Each bin = 2cs (20ms, one timestamp token).
Timestamp tokens mapping to silent bins get logits set to -INFINITY.

Simplified `whisper_stable_setup_filter`:
```c
bool whisper_stable_setup_filter(
    struct whisper_full_params & params,
    const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
    int64_t total_audio_cs,
    struct whisper_stable_ts_filter_ctx * filter_ctx);
```

No `mapping_processed_to_original` argument. No timeline complexity.

`whisper_stable_ts_filter_ctx` stays the same struct (mask + seek_cs + token_step_cs + wrapped callback).

---

## New `whisper-stable.cpp` Structure

```
whisper_stable_build_silence_map_from_pcm()   NEW — replaces VAD-based version
whisper_stable_build_silence_map()            KEEP — still useful for callers with VAD probs
whisper_stable_snap_timestamps()              REWRITE — correct stable-ts algorithm
whisper_stable_snap_segments()               SIMPLIFY — no VAD mapping, call snap then update segs
whisper_stable_get_gap_tokens()              KEEP as-is
whisper_stable_select_heads()               KEEP as-is
whisper_stable_silence_overlap_len()        KEEP as-is
whisper_stable_logits_filter_callback()     KEEP as-is (no changes needed)
whisper_stable_setup_filter()              SIMPLIFY — remove mapping_processed_to_original
```

Deleted entirely: `prepare`, `prepare_from_ctx`, `finalize`, `runtime` struct,
`build_timestamp_silence_mask`, `copy_vad_mapping`, `map_processed_to_original_time`.

---

## New `whisper.cpp` Wiring

In `whisper_full()` — before VAD stripping runs:

```c
// Save original PCM pointer before VAD might change it
const float * original_pcm = samples;
const int     original_n_samples = n_samples;
```

After VAD stripping (if any) and before `whisper_full_with_state`:

```c
std::vector<std::pair<int64_t,int64_t>> stable_silence;
whisper_stable_ts_filter_ctx stable_filter_ctx;

if (params.stable_timestamps) {
    // Build silence map from ORIGINAL PCM (before VAD stripping)
    stable_silence = whisper_stable_build_silence_map_from_pcm(
        original_pcm, original_n_samples, WHISPER_SAMPLE_RATE, /*min_silence_dur_cs=*/10);

    // Install constrained decoding filter
    const int64_t total_cs = (int64_t)((double)original_n_samples * 100.0 / WHISPER_SAMPLE_RATE);
    whisper_stable_setup_filter(params, stable_silence, total_cs, &stable_filter_ctx);
}
```

In the seek loop — update filter seek:
```c
if (params.stable_timestamps && params.logits_filter_callback == whisper_stable_logits_filter_callback) {
    whisper_stable_set_filter_seek(params.logits_filter_callback_user_data, seek);
}
```
(already correct in v1, keep as-is)

After `whisper_full_with_state` returns:
```c
if (params.stable_timestamps && !stable_silence.empty()) {
    whisper_stable_snap_segments(
        ctx,
        state->result_all,
        stable_silence,
        /*min_word_dur_cs=*/5,
        /*min_snap_silence_dur_cs=*/10);
}
```

No VAD mapping table touching. No `has_vad_segments` clearing. Clean.

---

## Files Changed vs v1

| File | Change |
|------|--------|
| `src/whisper-stable.cpp` | Delete ~350 lines, add ~120 lines. Net ~230 lines smaller. |
| `src/whisper-stable.h` | Delete ~10 declarations, keep ~8. |
| `src/whisper.cpp` | Simplify wiring. Remove VAD forcing. Add original PCM save. |
| Everything else | Unchanged. |

---

## TODO

- [ ] Delete removed functions from `whisper-stable.cpp`
- [ ] Delete removed declarations from `whisper-stable.h`
- [ ] Implement `whisper_stable_build_silence_map_from_pcm()`
- [ ] Rewrite `whisper_stable_snap_timestamps()` with correct algorithm
- [ ] Simplify `whisper_stable_snap_segments()` (remove VAD mapping path)
- [ ] Simplify `whisper_stable_setup_filter()` (remove mapping arg)
- [ ] Update `whisper_stable_runtime` struct → delete it
- [ ] Update `whisper.cpp` `whisper_full()` wiring (save original PCM, remove VAD forcing)
- [ ] Verify DTW gap padding row offset is correct (check `first_text_alignment_row` vs sot_sequence_length)
- [ ] Build and run verification against baseline with synthetic audio
- [ ] Confirm `start_in_silence` and `silence_overlaps` metrics improve vs baseline
