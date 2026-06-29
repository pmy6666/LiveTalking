# GPT-SoVITS Official Materials

Downloaded from the official GPT-SoVITS README demo asset.

- Source repo: https://github.com/RVC-Boss/GPT-SoVITS
- Source asset: https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb
- README label: "Unseen speakers few-shot fine-tuning demo"

Files:

- `gpt_sovits_fewshot_demo.mp4`: official demo video asset.
- `gpt_sovits_fewshot_demo_audio_16k_mono.wav`: extracted mono 16 kHz PCM WAV audio.
- `daily_texts_zh.json`: daily-life Chinese test sentences.
- `make_reference_clip.sh`: cut a short reference clip from the extracted official demo audio.
- `make_candidate_reference_clips.sh`: create several non-silent reference candidates for listening and selection.
- `generate_daily_tts.py`: call GPT-SoVITS api_v2 and generate test audio files.
- `run_daily_tts_test.sh`: convenience wrapper that creates a reference clip if needed and runs generation.
- `voice_refs_bilibili.json`: bilibili reference audio and transcript list.
- `run_bilibili_refs_test.py`: generate one output folder per bilibili reference voice.
- `run_bilibili_refs_test.sh`: convenience wrapper for bilibili reference evaluation.

Usage:

1. Start the GPT-SoVITS server from the LiveTalking project root:

   ```bash
   ./start_gpt_sovits_v2proplus.sh
   ```

2. Optional but recommended: create and listen to candidate reference clips:

   ```bash
   cd gpt_sovits_official_materials
   ./make_candidate_reference_clips.sh
   ```

   The candidates are written to:

   ```text
   gpt_sovits_official_materials/reference_candidates/
   ```

3. Run the daily-life TTS test:

   ```bash
   cd gpt_sovits_official_materials
   ./run_daily_tts_test.sh
   ```

3. Generated audio will be written to:

   ```text
   gpt_sovits_official_materials/generated_daily_tts/
   ```

Useful overrides:

```bash
REF_START=12 REF_DURATION=8 ./run_daily_tts_test.sh
OUT_DIR=generated_daily_tts_ref12 ./run_daily_tts_test.sh
REF_AUDIO=/path/to/clean_reference.wav PROMPT_TEXT="这里填写参考音频对应的准确文本" ./run_daily_tts_test.sh
SPEED_FACTOR=1.15 FRAGMENT_INTERVAL=0.05 ./run_daily_tts_test.sh
SERVER=http://127.0.0.1:9880 ./run_daily_tts_test.sh
```

Bilibili reference comparison:

```bash
cd gpt_sovits_official_materials
./run_bilibili_refs_test.sh
```

Each reference voice is written to a separate folder:

```text
gpt_sovits_official_materials/generated_bilibili_refs_tts/DongQing_6s/
gpt_sovits_official_materials/generated_bilibili_refs_tts/DongQing_6s_enhanced/
gpt_sovits_official_materials/generated_bilibili_refs_tts/Female/
gpt_sovits_official_materials/generated_bilibili_refs_tts/Female_enhanced/
gpt_sovits_official_materials/generated_bilibili_refs_tts/SaBeining/
gpt_sovits_official_materials/generated_bilibili_refs_tts/SaBeining_enhanced/
```

Run only selected references:

```bash
./run_bilibili_refs_test.sh --only DongQing_6s,Female,SaBeining_enhanced
```

Notes:

- The extracted WAV is the full demo audio track. It may contain multiple segments or speakers.
- For GPT-SoVITS reference audio, prefer a clean 5-10 second single-speaker clip with matching transcript.
- For few-shot fine-tuning, select clean speech clips and avoid background music, overlap, long silence, or mixed speakers.
- If `PROMPT_TEXT` is empty, the script can still run a smoke test, but clone quality may be unstable.
- The old `0s-8s` default clip is a bad reference: it contains about 3.18 seconds of leading silence and another long pause.
- The current default uses `REF_START=5 REF_DURATION=9` and transcript:
  `先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。`
- The default reference file is:
  `gpt_sovits_official_materials/reference_clips/official_ref_5s_14s.wav`
- `FRAGMENT_INTERVAL` controls pauses between generated fragments. Smaller values reduce sentence gaps; try `0.05` to `0.12`.
- `SPEED_FACTOR` controls speaking speed. Values above `1.0` are faster; try `1.08`, `1.12`, or `1.15`.
