# Stable Timestamps v2 — Validation Plan

Reuses existing verification infrastructure from v1 (`plans/stable-timestamps/`).
No new tooling needed.

---

## Setup

Build after rewrite:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target whisper-cli -j8
```

Models needed:
- `models/ggml-large-v3-turbo.bin` (whisper)
- No VAD model required anymore

---

## Step 1 — Generate Synthetic Audio

```bash
mkdir -p plans/stable-timestamps-v2/out
uv run plans/stable-timestamps/stable-timestamps_005.py \
  --output plans/stable-timestamps-v2/out/synth_5min.wav \
  --timeline-csv plans/stable-timestamps-v2/out/synth_5min_timeline.csv \
  --reference-txt plans/stable-timestamps-v2/out/synth_5min_reference.txt \
  --target-seconds 300 \
  --long-pause-seconds 20
```

Produces exact ground-truth silence timeline (CSV) used for all metric checks.

---

## Step 2 — Transcribe Baseline

```bash
./build/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps-v2/out/synth_5min.wav \
  -l en -ojf \
  -of plans/stable-timestamps-v2/out/baseline
```

---

## Step 3 — Transcribe Stable (no VAD model)

```bash
./build/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps-v2/out/synth_5min.wav \
  -l en --stable-timestamps -ojf \
  -of plans/stable-timestamps-v2/out/stable
```

No `--vad-model` flag — validates that VAD is no longer required.

---

## Step 4 — Compare

```bash
uv run plans/stable-timestamps/stable-timestamps_005_compare.py \
  --timeline-csv plans/stable-timestamps-v2/out/synth_5min_timeline.csv \
  --reference-txt plans/stable-timestamps-v2/out/synth_5min_reference.txt \
  --baseline-json plans/stable-timestamps-v2/out/baseline.json \
  --stable-json plans/stable-timestamps-v2/out/stable.json \
  --out-csv plans/stable-timestamps-v2/out/metrics_v2.csv
```

---

## Pass Criteria

| Metric | Baseline (v1 result) | Target |
|--------|---------------------|--------|
| `pct_start_in_silence` | 43.3% | < 15% |
| `n_overlap_silence` | 240 | < 100 |
| `wer` | 23.8% baseline / 2.6% stable | stable stays < 5% |
| `cer` | 21.6% baseline / 2.2% stable | stable stays < 5% |
| `hyp_to_ref_word_ratio` | ~1.0 | stays 0.95–1.05 (no hallucinations) |
| Runtime | 10.6s baseline / 30.2s stable | stable < 15s (no VAD overhead) |

Key check: **WER must not regress** vs stable v1. We're only fixing timestamps, not text.
Key check: **Runtime must improve** vs stable v1 (no VAD model load, no audio stripping).

---

## Sanity Checks

After step 3, manually inspect one segment from the JSON:
- Word timestamps must be strictly increasing
- No word duration < 50ms (min_word_dur_cs = 5)
- Segment t0/t1 must equal first/last word t0/t1
- No timestamp falls in a known long-silence region from the CSV

---

## Failure Modes to Watch

| Symptom | Likely cause |
|---------|-------------|
| WER shoots up vs baseline | Snapping algo moving boundaries too aggressively |
| No improvement in silence metrics | Silence map not detecting inter-word pauses (PCM energy threshold too high) |
| Word durations collapsing to min_word_dur | Snap is clipping word to wrong region (v1 bug still present) |
| Crash or empty output | PCM silence map builder bug |
| Runtime same as v1 | VAD model still being loaded somewhere |
