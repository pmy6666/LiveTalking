# LiveTalking EchoMimicV3 空闲态动画调整技术方案

## 1. 背景

当前 EchoMimicV3 已经可以在说话时播放生成出的数字人动画，但不说话时仍然播放固定图片。代码位置主要在：

- `avatars/echomimicv3_avatar.py`
- `_load_idle_frames()`：当前只读取 `ref_image_path`，返回 `[frame]`
- `_next_idle_frame()`：当前固定返回 `self.frame_list_cycle[0]`
- `EchoMimicV3Real.render()`：当 `_playback_frames` 为空时，调用 `_next_idle_frame()` 并推送静音音频

因此空闲态静止不是 WebRTC 或前端的问题，而是后端空闲帧源只有一帧，并且取帧逻辑没有轮播。

## 2. 目标

把“不说话时播放一张固定图片”调整为“不说话时播放一段自然循环的空闲动画”，例如轻微呼吸、眨眼、微表情、轻微头部摆动。

目标效果：

- 说话时继续播放 EchoMimicV3 生成视频。
- 等待生成或无人说话时播放空闲动画帧序列。
- 从说话动画回到空闲动画时不要突兀跳帧。
- 支持每个 avatar 独立配置自己的空闲素材。
- 如果没有空闲素材，保持现在的固定图兜底，不影响已有功能。

## 3. 推荐方案

推荐采用“预生成空闲帧序列 + 后端循环播放”的方案。

不要在实时空闲阶段调用 EchoMimicV3 动态生成，因为扩散模型延迟和显存占用都比较高。空闲动画本身可以离线生成或人工制作，运行时只需要读图片并按 25fps 推流，稳定、低延迟、易排查。

目录建议：

```text
data/avatars/wav2lip256_avatar1/
  echomimicv3/
    ref.png
    prompt.txt
    negative_prompt.txt
    idle_frames/
      000000.png
      000001.png
      000002.png
      ...
```

空闲帧建议：

- 帧率：25fps，和当前 `--fps 25` 保持一致。
- 时长：3 到 6 秒即可，即 75 到 150 帧。
- 动作：轻微呼吸、眨眼、眼神轻动、很小幅度点头。
- 分辨率：和参考图、EchoMimicV3 输出尺寸一致，建议 768x768。
- 编号：使用 `000000.png` 这种可排序命名。
- 循环：使用镜像循环，播放顺序为 `0,1,2,...,N-1,N-2,...,1`，避免首尾硬切。

## 4. 代码调整点

### 4.1 加载空闲帧

修改 `avatars/echomimicv3_avatar.py` 的 `_load_idle_frames()`。

当前逻辑：

```python
def _load_idle_frames(avatar_path: str, ref_image_path: str):
    frame = cv2.imread(ref_image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read reference image: {ref_image_path}")
    return [frame]
```

建议逻辑：

1. 优先读取 `data/avatars/<avatar_id>/echomimicv3/idle_frames/`。
2. 如果目录不存在或为空，退回当前参考图。
3. 所有空闲帧尺寸需要统一。如果尺寸不一致，加载时 resize 到参考图尺寸。
4. 加载完成后打印日志，方便确认是否真的启用了空闲动画。

伪代码：

```python
def _load_idle_frames(avatar_path: str, ref_image_path: str):
    ref_frame = cv2.imread(ref_image_path)
    if ref_frame is None:
        raise FileNotFoundError(f"Cannot read reference image: {ref_image_path}")

    idle_dir = os.path.join(avatar_path, "echomimicv3", "idle_frames")
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(idle_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        logger.info("EchoMimicV3 idle animation not found, fallback to ref image: %s", ref_image_path)
        return [ref_frame]

    target_h, target_w = ref_frame.shape[:2]
    frames = []
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            logger.warning("skip unreadable idle frame: %s", path)
            continue
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    if not frames:
        return [ref_frame]

    logger.info("EchoMimicV3 idle animation loaded: frames=%d dir=%s", len(frames), idle_dir)
    return frames
```

### 4.2 空闲态轮播

修改 `EchoMimicV3Real._next_idle_frame()`。

当前逻辑：

```python
def _next_idle_frame(self):
    return self.frame_list_cycle[0]
```

建议逻辑：

```python
def _next_idle_frame(self):
    length = len(self.frame_list_cycle)
    if length <= 1:
        return self.frame_list_cycle[0]
    frame = self.frame_list_cycle[mirror_index(length, self._idle_index)]
    self._idle_index += 1
    return frame
```

这样空闲时每次渲染循环都会取下一帧，并且使用已有 `mirror_index()` 做自然往返循环。

### 4.3 说话与空闲的过渡

当前 EchoMimicV3 的 `render()` 在 `_playback_frames` 为空时会立即切回 `_next_idle_frame()`，从生成动画最后一帧跳到空闲动画当前帧时可能突兀。

建议加一个轻量过渡：

- 记录最后一帧说话画面：`self._last_playback_frame`
- 第一次从 `PLAYING_GENERATED` 回到 `IDLE_STATIC` 时，做 3 到 5 帧 `cv2.addWeighted()` 淡入
- 过渡完成后正常播放空闲帧序列

实现上可以先做最小版：只在状态从 `PLAYING_GENERATED` 变成 `IDLE_STATIC` 时，将 `self._idle_index = 0`，让空闲动画从第 0 帧开始，避免落在随机帧上。后续再加淡入。

### 4.4 配置项

建议增加命令行参数，方便开关和调试：

```python
parser.add_argument('--idle_animation', action='store_true',
                    help='enable avatar idle frame sequence if available')
parser.add_argument('--idle_animation_dir', type=str, default='',
                    help='optional idle frame directory, defaults to data/avatars/<avatar_id>/echomimicv3/idle_frames')
parser.add_argument('--idle_reset_on_return', action='store_true',
                    help='reset idle animation index when generated speech playback ends')
```

如果想保持简单，第一阶段可以不加参数，直接约定 `idle_frames/` 存在就启用，不存在就回退固定图。

## 5. 空闲素材制作方案

### 方案 A：从真人短视频抽帧

适合质量最高的场景。

1. 录制一段 3 到 6 秒人物不说话视频，保持正脸、光照稳定。
2. 用 ffmpeg 抽帧：

```bash
ffmpeg -i idle.mp4 -vf fps=25,scale=768:768 data/avatars/wav2lip256_avatar1/echomimicv3/idle_frames/%06d.png
```

3. 选取首帧或最自然的一帧作为 `ref.png`，保证说话生成和空闲动画身份、姿态一致。

### 方案 B：用 EchoMimicV3 离线生成空闲短片

适合没有真人空闲视频的场景。

1. 使用同一张 `ref.png`。
2. 准备一段静音或极低能量音频。
3. prompt 描述轻微自然动作，例如“the person is idle, breathing naturally, blinking occasionally, subtle head movement”。
4. 离线生成视频后抽帧到 `idle_frames/`。

注意：不要每次在线空闲时生成，应该离线生成一次，运行时只读帧。

### 方案 C：后处理生成轻微动效

适合临时演示。

可以基于单张图做非常轻微的仿射变换、眨眼贴图或面部关键点 warping。但质量通常不如真实空闲视频，容易出现“整张图漂移”的感觉。建议只作为兜底方案。

## 6. 与现有 customvideo_config 的关系

项目已有 `--customvideo_config` 和 `set_audiotype` 机制，可以播放自定义图片序列和音频。但它更像“指定动作状态”机制，当前 EchoMimicV3 的 `_next_custom_frame_and_audio()` 还要求 `audiotype > 1` 且有 `custom_audio_index`，不适合作为默认静默空闲态。

因此推荐不要把默认空闲动画塞进 `customvideo_config`，而是在 EchoMimicV3 avatar 自己的 `idle_frames/` 里处理。这样职责更清楚：

- 默认无人说话：播放 `idle_frames/`
- 业务指定动作：继续使用 `customvideo_config` / `set_audiotype`
- 说话内容：播放 EchoMimicV3 生成结果或 choice cache

## 7. 实施步骤

第一阶段，最小可用：

1. 在 `data/avatars/wav2lip256_avatar1/echomimicv3/idle_frames/` 放入 75 到 150 张空闲帧。
2. 修改 `_load_idle_frames()`，优先加载 `idle_frames/`。
3. 修改 `_next_idle_frame()`，从固定第 0 帧改为按 `_idle_index` 循环。
4. 启动现有命令验证：

```bash
../envs/livetalking/bin/python app.py \
  --transport webrtc \
  --model echomimicv3 \
  --avatar_id wav2lip256_avatar1 \
  --tts gpt-sovits \
  --TTS_SERVER http://127.0.0.1:9880 \
  --TTS_MEDIA_TYPE wav \
  --GPT_SOVITS_STREAMING_MODE 2 \
  --REF_FILE DongQing_6s.wav \
  --REF_TEXT "那种快乐常常像一场梦，电影陪伴我们长大" \
  --max_session 1 \
  --echomimicv3_repo third_party/echomimic_v3 \
  --echomimicv3_model_dir EchoMimicV3 \
  --echomimicv3_base_model_dir Wan2.1-Fun-1.3B-InP \
  --echomimicv3_wav2vec_dir chinese-wav2vec2-base \
  --echomimicv3_sample_size 768 768 \
  --echomimicv3_video_length 201 \
  --echomimicv3_num_steps 20 \
  --echomimicv3_guidance_scale 5.5 \
  --echomimicv3_audio_guidance_scale 4.0 \
  --echomimicv3_gpu_memory_mode model_cpu_offload \
  --listenport 8010
```

第二阶段，体验优化：

1. 增加说话结束到空闲动画的 3 到 5 帧淡入。
2. 增加 `--idle_animation_dir` 参数，方便不同角色复用同一套空闲帧。
3. 在日志里输出当前空闲帧数量、分辨率、播放模式。
4. 如果空闲帧过多，限制最大加载数量或改为懒加载，避免内存占用过大。

## 8. 验证清单

启动后检查日志：

- 出现 `EchoMimicV3 idle animation loaded: frames=...`
- 没有出现大量 `skip unreadable idle frame`
- WebRTC 页面无人说话时画面在动
- 说话期间音画仍同步
- 说话结束后能回到空闲动画
- 连续多轮对话后空闲动画不会卡住
- `flush_talk` 后能回到空闲态

性能检查：

- 空闲阶段 GPU 不应有明显新增负载
- CPU 只做图片推流和音频静音帧推送
- WebRTC 视频队列 `buffer_size` 不应持续增长

## 9. 风险与处理

- 空闲帧和生成视频尺寸不同：加载时统一 resize。
- 空闲首尾跳变：使用 `mirror_index()` 往返循环，或制作首尾可无缝循环素材。
- 人物身份和姿态不一致：空闲帧必须来自同一角色、同一构图，最好和 `ref.png` 同源。
- 说话结束跳变明显：增加淡入过渡，或让 `idle_frames/000000.png` 尽量接近说话生成的默认姿态。
- 内存占用偏高：150 张 768x768 BGR 图片约 250MB，建议控制在 75 到 150 帧；需要更长空闲效果时用镜像循环，不要无限加帧。

## 10. 结论

最合适的调整方式是：为每个 EchoMimicV3 avatar 增加 `echomimicv3/idle_frames/` 空闲帧序列，并把后端从“固定返回第 0 帧”改成“按帧循环播放”。这条路线改动小、运行稳定、不会增加实时生成延迟，也能保留当前固定图片作为兜底。

