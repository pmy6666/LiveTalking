# Two-Stage 选择视频缓存接入说明

## 当前运行时的视频缓存机制

当前选择模式的视频缓存命中逻辑主要在这些文件里：

- `choice/orchestrator.py`
- `choice/echomimicv3_cache.py`
- `avatars/echomimicv3_avatar.py`

当用户点击某个选择项后，`ChoiceOrchestrator._play_or_enqueue()` 会先判断当前数字人是否使用：

```text
model == echomimicv3
```

如果是，就尝试读取 EchoMimicV3 的选择视频缓存。

当前运行时真正识别的缓存结构是：

```text
cache/choice_echomimicv3/<tree_id>/manifest.json
cache/choice_echomimicv3/<tree_id>/<cache_key>/meta.json
cache/choice_echomimicv3/<tree_id>/<cache_key>/audio.npy
cache/choice_echomimicv3/<tree_id>/<cache_key>/frames.npz
```

也就是说，当前运行时不会直接读取 `.mp4` 文件作为选择视频缓存。

## 当前命中缓存需要满足什么条件

运行时会先从：

```text
cache/choice_echomimicv3/<tree_id>/manifest.json
```

里找到当前 `node_id` 对应的 `ready` 缓存项，然后继续检查对应目录里是否存在：

```text
meta.json
audio.npy
frames.npz
```

随后 `choice/echomimicv3_cache.py` 会校验 metadata 是否和当前运行时完全匹配。关键字段包括：

```text
schema_version
node_id
answer_text
model
tts
tts_ref_file
tts_ref_file_sha1
tts_ref_text
sample_size
fps
num_steps
guidance_scale
audio_guidance_scale
transformer_path
weight_dtype
ref_image_path
ref_image_sha1
```

只要其中有一个字段不一致，当前逻辑就会认为视频缓存不兼容，然后回退到音频缓存或实时生成。

## 现在 Two-Stage 脚本生成了什么

当前新的 two-stage 预生成脚本是：

```text
scripts/two_stage_pre/precompute_choice_two_stage.py
```

它会复用这个配置里的参数：

```text
two_stage/configs/two_stage_avatar7_dongqing.yaml
```

并分别使用：

```text
male   -> assets/avatars/avatar6.png + cache/choice_audio_wav/male/*.wav
female -> assets/avatars/avatar7.png + cache/choice_audio_wav/female/*.wav
```

生成 two-stage 视频。

输出结构是：

```text
cache/choice_two_stage/<voice>/<node_id>/config.yaml
cache/choice_two_stage/<voice>/<node_id>/stage1_echomimicv3/...
cache/choice_two_stage/<voice>/<node_id>/stage2_latentsync/...
cache/choice_two_stage/<voice>/<node_id>/final/<node_id>.mp4
cache/choice_two_stage/manifest.json
```

这些 `.mp4` 很适合离线检查和作为高质量预渲染资产，但它们目前不能被现有的 `video_cache_hit=true` 路径直接命中。

## 为什么现在不能直接命中

原因很简单：两边格式不同。

当前运行时需要：

```text
frames.npz + audio.npy + meta.json
```

但 two-stage 脚本生成的是：

```text
final/<node_id>.mp4
```

因此，除非增加转换步骤或修改运行时逻辑，否则当前选择模式不会自动使用这些 two-stage mp4。

## 推荐接入方案

### 方案 A：把 Two-Stage MP4 转成现有缓存格式

这是最推荐的方案，因为可以不改运行时播放逻辑。

每个 two-stage 视频生成后，做一次导入转换：

1. 把最终 `.mp4` 解码成帧。
2. 读取对应节点的 `.wav` 音频。
3. 把音频转成 16 kHz、float32。
4. 对齐音频长度到：

```text
frames_count * 2 * avatar_session.chunk
```

5. 写入：

```text
cache/choice_echomimicv3/default_choice_tree/<cache_key>/meta.json
cache/choice_echomimicv3/default_choice_tree/<cache_key>/audio.npy
cache/choice_echomimicv3/default_choice_tree/<cache_key>/frames.npz
```

6. 更新：

```text
cache/choice_echomimicv3/default_choice_tree/manifest.json
```

把对应 `node_id` 的缓存项标记为 `ready`。

这样现有的 `ChoiceEchoMimicV3CacheStore.get()` 和 `play_cached_video_segment()` 就可以继续按原逻辑工作。

需要注意：`cache_key` 和 `meta.json` 必须按照当前 `ChoiceEchoMimicV3CacheStore.build_params()` / `is_compatible()` 需要的字段生成。运行时使用的头像图、TTS 参考音频、参考文本、EchoMimicV3 参数和选择树文本都必须匹配。

### 方案 B：新增 MP4 缓存播放路径

另一种方案是改运行时，让选择模式直接读取：

```text
cache/choice_two_stage/<voice>/<node_id>/final/<node_id>.mp4
```

这样 two-stage 输出可以直接使用，但需要新增一套 mp4 缓存读取和播放逻辑。

大致需要：

- 运行时根据当前数字人或音色判断使用 `male` 还是 `female`
- 根据当前 `node_id` 找到对应 mp4
- 解码 mp4 成帧和音频
- 或者新增一个可以直接播放 mp4 片段的数字人接口

这个方案概念上直观，但需要改运行时代码。

### 方案 C：暂时保留两套缓存

短期也可以保持两套缓存：

- `cache/choice_audio/`：选择模式的音频缓存
- `cache/choice_two_stage/`：离线生成的高质量 two-stage 视频资产

这种方式风险最低，但不会让当前运行时自动命中 two-stage 视频。

## 当前可执行流程

先生成 TTS 音频和可试听 wav：

```bash
cd .
../envs/livetalking/bin/python scripts/tts_precompute/precompute_choice_audio_cache.py \
  --tree_id default_choice_tree \
  --force \
  --export_wav
```

然后生成 two-stage 视频：

```bash
cd .
../envs/livetalking/bin/python scripts/two_stage_pre/precompute_choice_two_stage.py \
  --python ../envs/livetalking/bin/python
```

生成结果可以在这里查看：

```text
cache/choice_two_stage/male/<node_id>/final/<node_id>.mp4
cache/choice_two_stage/female/<node_id>/final/<node_id>.mp4
```

如果要让当前运行时命中这些视频，需要继续实现“方案 A”的导入转换脚本。

## 建议下一步脚本

建议新增一个导入脚本，例如：

```text
scripts/two_stage_pre/import_two_stage_to_choice_cache.py
```

这个脚本负责：

```text
- 读取 cache/choice_two_stage/manifest.json
- female 映射到 avatar7.png 和 DongQing_6s.wav
- male 映射到 avatar6.png 和 SaBeining.wav
- 解码 final mp4 里的视频帧
- 读取对应 wav 音频
- 生成兼容 ChoiceEchoMimicV3CacheStore 的 meta.json
- 写入 frames.npz/audio.npy/meta.json
- 更新 cache/choice_echomimicv3/default_choice_tree/manifest.json
```

完成后，当前运行时在选择节点时就有机会返回：

```json
{
  "video_cache_hit": true,
  "cache_mode": "echomimicv3_precomputed"
}
```

## 命中前检查清单

如果希望某个节点命中视频缓存，需要确认：

```text
tree_id == default_choice_tree
node_id 和当前选择目标节点一致
answer_text/tts_text 在缓存生成后没有变化
运行时 model == echomimicv3
运行时头像参考图和缓存 metadata 一致
运行时 TTS 参考音频路径和参考文本与缓存 metadata 一致
运行时 sample_size/fps/num_steps/guidance/audio_guidance/weight_dtype 与缓存 metadata 一致
manifest 里对应缓存项 status == ready
meta.json、audio.npy、frames.npz 都存在
```

只要有任意一项不一致，当前代码就会跳过视频缓存，回退到音频缓存或实时生成。
