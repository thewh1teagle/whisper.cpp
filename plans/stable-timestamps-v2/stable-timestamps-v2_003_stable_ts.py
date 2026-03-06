# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "stable-ts",
#   "faster-whisper",
# ]
# ///

"""
Run stable-ts on the same synthetic audio and output whisper.cpp-compatible JSON
for side-by-side comparison with our --stable-timestamps implementation.

Usage:
    uv run plans/stable-timestamps-v2/stable-timestamps-v2_003_stable_ts.py \
      --audio plans/stable-timestamps-v2/out/synth_5min.wav \
      --output-json plans/stable-timestamps-v2/out/stable_ts_ref.json \
      --model large-v3-turbo
"""

import argparse
import json
from pathlib import Path

import stable_whisper


def fmt_ts(t: float) -> str:
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = int(t % 60)
    ms = int(round((t % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",       required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model",       default="large-v3-turbo")
    args = parser.parse_args()

    print(f"Loading faster-whisper model: {args.model}")
    model = stable_whisper.load_faster_whisper(args.model)

    print(f"Transcribing: {args.audio}")
    result = model.transcribe(args.audio, word_timestamps=True)

    transcription = []
    for seg in result.segments:
        tokens = []
        for w in (seg.words or []):
            t0_ms = int(round(w.start * 1000))
            t1_ms = int(round(w.end   * 1000))
            tokens.append({
                "text": w.word,
                "timestamps": {
                    "from": fmt_ts(w.start),
                    "to":   fmt_ts(w.end),
                },
                "offsets": {
                    "from": t0_ms,
                    "to":   t1_ms,
                },
            })
        transcription.append({
            "offsets": {
                "from": int(seg.start * 1000),
                "to":   int(seg.end   * 1000),
            },
            "text": seg.text,
            "tokens": tokens,
        })

    out = Path(args.output_json)
    out.write_text(json.dumps({"transcription": transcription}, indent=2))
    print(f"Saved {len(transcription)} segments → {out}")


if __name__ == "__main__":
    main()
