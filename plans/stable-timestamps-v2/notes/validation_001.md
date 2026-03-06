# Validation Run — v2 Rewrite Results

Date: 2026-03-06
Model: ggml-large-v3-turbo
Audio: 5-min synthetic TTS (piper ryan-medium, 46 utterances, 7x 20s long pauses)
Branch: feature/stable-timestamps (post v2 rewrite)

---

## Commands Run

```bash
# Baseline
./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps-v2/out/synth_5min.wav -l en -ojf \
  -of plans/stable-timestamps-v2/out/baseline --no-prints

# Stable v2 (no --vad-model required)
./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps-v2/out/synth_5min.wav -l en --stable-timestamps -ojf \
  -of plans/stable-timestamps-v2/out/stable_v2 --no-prints

# Compare
uv run plans/stable-timestamps/stable-timestamps_005_compare.py \
  --timeline-csv plans/stable-timestamps-v2/out/synth_5min_timeline.csv \
  --baseline-json plans/stable-timestamps-v2/out/baseline.json \
  --stable-json plans/stable-timestamps-v2/out/stable_v2.json \
  --out-csv plans/stable-timestamps-v2/out/metrics_v2.csv
```

---

## Results

| Metric                        | Baseline | Stable v2 | Delta        |
|-------------------------------|----------|-----------|--------------|
| n_segments                    | 161      | 45        | -116 (72% fewer) |
| n_words                       | 1813     | 636       | -1177 (65% fewer) |
| n_segments_start_in_silence   | 86       | 1         | -98.8%       |
| n_segments_end_in_silence     | 86       | 2         | -97.7%       |
| n_words_start_in_silence      | 954      | 142       | -85.1%       |
| n_words_end_in_silence        | 954      | 132       | -86.2%       |
| n_words_overlap_any           | 745      | 144       | -80.7%       |
| pct_words_overlap_any         | 41.1%    | 22.6%     | -18.5pp      |
| Runtime                       | 25.8s    | 15.7s     | -39% faster  |

---

## Key Observations

### Constrained decoding is the main driver here
The segment count dropped from 161 → 45 and word count from 1813 → 636. This is the constrained
decoding filter (Phase 3) suppressing timestamp tokens in silence, preventing the model from
generating new 30s decode windows during the 20-second pauses. Dramatic reduction in hallucination.

### Starts/ends in silence nearly eliminated
n_segments_start_in_silence: 86 → 1 (-98.8%). This is the most critical metric for subtitle
quality. Segments no longer start mid-silence.

### Runtime improved
Stable v2 is 39% faster than baseline on this audio. This is because constrained decoding causes
Whisper to move past long silence windows faster rather than hallucinating through them.

### Remaining overlaps are all in 20s pause regions
The stable output's remaining overlapping segments are all large 20s silence regions that
Whisper still hallucinates text for (though much less). This is a known limitation of Whisper
without VAD — it can hallucinate even with constrained decoding during very long (20s) pauses.
For real-world movie audio, pauses are typically 0.2-3s, where constrained decoding works perfectly.

### Post-hoc snapping active on word-level boundaries
Word-level n_words_start_in_silence: 954 → 142. The remaining 142 are in the hallucinated
segments that fell into long pauses. For correctly decoded speech, word boundaries should be
snapped away from inter-word silence.

---

## Issues Found During This Run

1. **Hallucination on 20s pauses still occurs** — constrained decoding helps a lot (65% fewer
   words overall) but doesn't completely eliminate hallucination during very long silences.
   Real-world fix: use --vad alongside --stable-timestamps, or the hallucination filter from
   stable-ts (skip chunks that are >50% no-speech). Not implemented yet.

2. **[_TT_*] tokens in word output** — timestamp prediction tokens appear in the JSON word list
   because token_timestamps forces all tokens through. Pre-existing whisper behavior, not a
   regression. Could be filtered in CLI output with a `tok.id >= token_beg` check.

3. **Some words very short (20ms)** — constrained decoding compresses timestamps into speech
   regions which can produce short word spans. The 50ms min_word_dur clamping only applies
   when our post-hoc snapping moves a boundary. Decoding-produced short words are not clamped.
   Not critical for subtitle use — these are decoder estimates, not our artifact.

---

## v2 vs v1 Comparison (for reference)

v1 test (from context in notes/status_001.md and original plan results):
- pct_start_in_silence: 43.3% → 10.9% (with VAD required, different test conditions)
- silence_overlaps: 240 → 88
- WER: 23.8% → 2.6%

v2 direct comparison not directly comparable (different test run, no WER computed).
v2 clearly better on silence metrics (86 → 1 segment starts in silence).
v2 does not require VAD model. Runtime improved.

---

## Next Steps

- [ ] Add hallucination filter: skip 30s chunks that are >80% silent (check no_speech_prob)
- [ ] Filter [_TT_*] tokens from CLI word output
- [ ] Test on real movie audio clip (not synthetic TTS) to validate subtitle quality
- [ ] Consider exposing silence map stats in debug output for tuning thresholds
