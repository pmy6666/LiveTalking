# LatentSync 1.6 Test

默认测试使用：

- 视频：`third_party/LatentSync/assets/demo1_video.mp4`
- 音频：`gpt_sovits_official_materials/generated_bilibili_refs_tts/DongQing_6s/01_morning_breakfast.wav`
- 输出：`LatentSync_test/generated_videos_dongqing_sync`
- 默认同步优先参数：`--guidance-scale 2.2 --inference-steps 30`

先检查依赖：

```bash
cd .
LatentSync_test/run_latentsync_dongqing_batch.sh --skip-dep-check --only 01_morning_breakfast
```

正常运行：

```bash
cd .
LatentSync_test/run_latentsync_dongqing_batch.sh --only 01_morning_breakfast
```

批量运行整个董卿 TTS 目录：

```bash
cd .
LatentSync_test/run_latentsync_dongqing_batch.sh --all
```

更强同步但可能更抖：

```bash
cd .
LatentSync_test/run_latentsync_dongqing_batch.sh --guidance-scale 2.6 --inference-steps 40
```

如果强同步版本出现脸部抖动或嘴部变形，回退到更稳的参数：

```bash
cd .
LatentSync_test/run_latentsync_dongqing_batch.sh --guidance-scale 1.8 --inference-steps 30
```

LatentSync 1.6 官方建议至少约 18GB 显存。若显存不够，可以先保留默认 `--enable-deepcache`，或减少同时运行的其它 GPU 进程。
