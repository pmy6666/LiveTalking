#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_AUDIO="${SOURCE_AUDIO:-$SCRIPT_DIR/gpt_sovits_fewshot_demo_audio_16k_mono.wav}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/reference_candidates}"

mkdir -p "$OUT_DIR"

# Candidates are cut from non-silent spans detected in the official demo audio.
# Listen to them and keep only clips that are single-speaker, clean, and match a transcript.
declare -a CLIPS=(
  "03.2 1.6 candidate_01_short_after_intro.wav"
  "09.6 2.0 candidate_02_short.wav"
  "12.2 1.7 candidate_03_short.wav"
  "14.6 1.4 candidate_04_short.wav"
  "18.4 1.2 candidate_05_short.wav"
  "23.1 1.3 candidate_06_short.wav"
  "28.9 1.3 candidate_07_short.wav"
  "34.1 2.3 candidate_08_medium.wav"
  "45.6 4.7 candidate_09_good_span.wav"
  "52.8 5.7 candidate_10_good_span.wav"
  "61.2 5.7 candidate_11_good_span.wav"
  "69.9 4.4 candidate_12_good_span.wav"
  "76.3 4.5 candidate_13_good_span.wav"
  "81.4 3.3 candidate_14_medium.wav"
)

for item in "${CLIPS[@]}"; do
  read -r start duration name <<<"$item"
  ffmpeg -y -hide_banner -loglevel error \
    -ss "$start" \
    -t "$duration" \
    -i "$SOURCE_AUDIO" \
    -af "afade=t=in:st=0:d=0.02,afade=t=out:st=$(awk "BEGIN { print ($duration > 0.04) ? $duration - 0.02 : 0 }"):d=0.02" \
    -ac 1 \
    -ar 16000 \
    -c:a pcm_s16le \
    "$OUT_DIR/$name"
  echo "$OUT_DIR/$name"
done
