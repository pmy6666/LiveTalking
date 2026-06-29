# LatentSync UNet Three-Stage Code Report

本文档排查 `latentsync/models/unet.py` 中 UNet 阶段与论文图中三类模块的对应关系：

- 白色：`conv + self-attn`
- 蓝色：`cross-attn`
- 灰色：`temporal layer`

结论先行：LatentSync 的 UNet 主体由 `UNet3DConditionModel` 负责搭建 encoder / mid / decoder；每个 CrossAttn block 的实际执行顺序是 `ResnetBlock3D(conv)` -> `Transformer3DModel(self-attn + audio cross-attn + FFN)` -> `motion_module(temporal layer)`。非 CrossAttn block 没有 audio cross-attn，执行顺序是 `ResnetBlock3D(conv)` -> `motion_module(temporal layer)`。

## 1. UNet 总体入口和 block 拼装

| 位置 | 文件与行号 | 作用 |
| --- | --- | --- |
| UNet 类定义 | `latentsync/models/unet.py:39` | `UNet3DConditionModel` 是 LatentSync 的 UNet 主类。 |
| 模型关键配置参数 | `latentsync/models/unet.py:43-84` | 定义 `down_block_types`、`up_block_types`、`cross_attention_dim`、`use_motion_module`、`add_audio_layer` 等。 |
| 输入卷积 | `latentsync/models/unet.py:92` | `conv_in = InflatedConv3d(...)`，把 channel-wise concatenate 后的输入映射到第一个 block channel。 |
| down blocks 构造 | `latentsync/models/unet.py:120-154` | 逐层调用 `get_down_block(...)`，并传入 cross-attn、motion module、audio layer 参数。 |
| mid block 构造 | `latentsync/models/unet.py:156-176` | 构造 `UNetMidBlock3DCrossAttn`。 |
| up blocks 构造 | `latentsync/models/unet.py:183-227` | 逐层调用 `get_up_block(...)`，并传入 skip connection、cross-attn、motion module、audio layer 参数。 |
| 输出卷积 | `latentsync/models/unet.py:230-241` | `conv_norm_out`、`conv_act`、`conv_out` 输出预测噪声。 |
| UNet forward 主流程 | `latentsync/models/unet.py:312-472` | 完成 timestep embedding、`conv_in`、down、mid、up、`conv_out`。 |
| down forward 调用 | `latentsync/models/unet.py:398-413` | 如果 block 有 cross-attn，就传入 `encoder_hidden_states` 和 `attention_mask`。 |
| mid forward 调用 | `latentsync/models/unet.py:423-426` | mid block 接收 audio embedding 作为 `encoder_hidden_states`。 |
| up forward 调用 | `latentsync/models/unet.py:434-463` | decoder block 接收 skip states 和 audio embedding。 |
| post-process | `latentsync/models/unet.py:464-467` | norm、activation、output conv。 |

默认 stage2 配置中的 UNet 结构：

- `configs/unet/stage2.yaml:53`：`add_audio_layer: true`，开启 audio cross-attn。
- `configs/unet/stage2.yaml:57`：`cross_attention_dim: 384`，与 Whisper audio embedding 维度对应。
- `configs/unet/stage2.yaml:58-64`：down blocks 为 3 个 `CrossAttnDownBlock3D` + 1 个 `DownBlock3D`。
- `configs/unet/stage2.yaml:65`：mid block 为 `UNetMidBlock3DCrossAttn`。
- `configs/unet/stage2.yaml:66-72`：up blocks 为 1 个 `UpBlock3D` + 3 个 `CrossAttnUpBlock3D`。
- `configs/unet/stage2.yaml:85-99`：开启 motion module，也就是 temporal layer 的代码路径。

## 2. UNet 输入与 audio embedding 来源

训练时：

| 位置 | 文件与行号 | 作用 |
| --- | --- | --- |
| 加载 UNet | `scripts/train_unet.py:126-130` | `UNet3DConditionModel.from_pretrained(...)` 从 config 和 ckpt 构造模型。 |
| stage2 只训练指定模块 | `scripts/train_unet.py:148-158` | 如果 `use_motion_module` 为 true，只放开 `trainable_modules` 中包含的参数。 |
| audio embedding 裁剪 | `scripts/train_unet.py:266-284` | 从 audio encoder 取 `(B, 16, 50, 384)` 的 audio embedding。 |
| VAE latent / mask / ref latent | `scripts/train_unet.py:288-316` | 得到 gt、masked、reference latents 和 masks。 |
| 加噪 latent | `scripts/train_unet.py:318-342` | 构造 noisy gt latents。 |
| channel-wise concatenate | `scripts/train_unet.py:352` | `torch.cat([noisy_gt_latents, masks, masked_latents, ref_latents], dim=1)`，对应图中的 channel-wise concatenate。 |
| 调用 UNet | `scripts/train_unet.py:356-357` | `unet(unet_input, timesteps, encoder_hidden_states=audio_embeds)`。 |

推理时：

| 位置 | 文件与行号 | 作用 |
| --- | --- | --- |
| 加载 UNet | `scripts/inference.py:120-124` | 从 config 和 inference ckpt 构造 UNet。 |
| Whisper feature to chunks | `latentsync/pipelines/lipsync_pipeline.py:569-570` | 生成按视频帧对齐的 Whisper chunks。 |
| audio embeds batch | `latentsync/pipelines/lipsync_pipeline.py:595-600` | 堆叠当前 chunk 的 audio embedding；CFG 时拼接 null audio embedding。 |
| mask / masked image latent | `latentsync/pipelines/lipsync_pipeline.py:277-304` | 准备 mask latent 和 masked image latent。 |
| reference latent | `latentsync/pipelines/lipsync_pipeline.py:306-313` | 编码 reference frames。 |
| denoise loop 拼接输入 | `latentsync/pipelines/lipsync_pipeline.py:630-644` | `latents + mask_latents + masked_image_latents + ref_latents` 后送入 UNet。 |

## 3. 白色部分：conv + self-attn

### 3.1 Conv / ResNet 部分

白色中的 conv 主要由 `InflatedConv3d`、`ResnetBlock3D`、downsample / upsample conv 组成。`InflatedConv3d` 继承自 `nn.Conv2d`，通过 reshape 把视频维度 `f` 合并进 batch，对每帧做 2D conv，再 reshape 回 `(b, c, f, h, w)`。

| 位置 | 文件与行号 | 作用 |
| --- | --- | --- |
| InflatedConv3d | `latentsync/models/resnet.py:10-18` | 对 5D video tensor 做 frame-wise 2D conv。 |
| InflatedGroupNorm | `latentsync/models/resnet.py:21-29` | 对 5D video tensor 做 frame-wise group norm。 |
| ResnetBlock3D 定义 | `latentsync/models/resnet.py:104-180` | UNet block 内的主要 conv/residual 单元。 |
| ResnetBlock3D conv1 | `latentsync/models/resnet.py:142` | 第一层 3x3 `InflatedConv3d`。 |
| ResnetBlock3D conv2 | `latentsync/models/resnet.py:167` | 第二层 3x3 `InflatedConv3d`。 |
| ResnetBlock3D shortcut | `latentsync/models/resnet.py:176-180` | channel 不一致时用 1x1 `InflatedConv3d` 做 shortcut。 |
| ResnetBlock3D forward | `latentsync/models/resnet.py:182-223` | norm -> act -> conv1 -> time embedding -> norm -> act -> dropout -> conv2 -> residual add。 |
| downsample conv | `latentsync/models/resnet.py:78-101` | `Downsample3D` 使用 stride=2 的 `InflatedConv3d`。 |
| upsample conv | `latentsync/models/resnet.py:32-75` | `Upsample3D` 先 nearest 插值，再 `InflatedConv3d`。 |
| UNet input conv | `latentsync/models/unet.py:92` 和 `latentsync/models/unet.py:395-396` | UNet 开头的 `conv_in`。 |
| UNet output conv | `latentsync/models/unet.py:230-241` 和 `latentsync/models/unet.py:464-467` | UNet 末尾的 `conv_out`。 |

Conv block 在各类 UNet block 中的构造位置：

| block 类型 | 文件与行号 | 说明 |
| --- | --- | --- |
| `UNetMidBlock3DCrossAttn` | `latentsync/models/unet_blocks.py:183-198`、`latentsync/models/unet_blocks.py:227-241` | mid block 开头有一个 resnet，attention 后还有一个 resnet。 |
| `CrossAttnDownBlock3D` | `latentsync/models/unet_blocks.py:299-315` | 每层先构造 `ResnetBlock3D`。 |
| `DownBlock3D` | `latentsync/models/unet_blocks.py:435-450` | 无 cross-attn 的 down block，仅 resnet + optional temporal。 |
| `CrossAttnUpBlock3D` | `latentsync/models/unet_blocks.py:555-573` | 先拼 skip connection，再进 `ResnetBlock3D`。 |
| `UpBlock3D` | `latentsync/models/unet_blocks.py:694-711` | 无 cross-attn 的 up block，仅 resnet + optional temporal。 |

Conv block 在 forward 中的执行位置：

| block 类型 | 文件与行号 | 执行顺序 |
| --- | --- | --- |
| `UNetMidBlock3DCrossAttn` | `latentsync/models/unet_blocks.py:247-258` | `resnets[0]` -> attention -> optional motion -> resnet。 |
| `CrossAttnDownBlock3D` | `latentsync/models/unet_blocks.py:362-397` | resnet -> attention -> optional motion。 |
| `DownBlock3D` | `latentsync/models/unet_blocks.py:481-506` | resnet -> optional motion。 |
| `CrossAttnUpBlock3D` | `latentsync/models/unet_blocks.py:620-660` | concat skip -> resnet -> attention -> optional motion。 |
| `UpBlock3D` | `latentsync/models/unet_blocks.py:741-771` | concat skip -> resnet -> optional motion。 |

### 3.2 Self-attn 部分

白色中的 self-attn 在 `Transformer3DModel` 的 `BasicTransformerBlock.attn1` 中实现。注意这里的 self-attn 是 spatial self-attn：代码先把 `(b, c, f, h, w)` reshape 成 `(b*f, h*w, c)`，每一帧内部的 spatial tokens 做 attention。

| 位置 | 文件与行号 | 作用 |
| --- | --- | --- |
| `Transformer3DModel` 构造 | `latentsync/models/attention.py:23-81` | 定义 norm、输入投影、`BasicTransformerBlock` 列表、输出投影。 |
| video reshape 到 spatial tokens | `latentsync/models/attention.py:82-99` | `(b, c, f, h, w)` -> `(b*f, h*w, inner_dim)`。 |
| transformer block forward | `latentsync/models/attention.py:101-108` | 逐个执行 `BasicTransformerBlock`。 |
| reshape 回 video tensor | `latentsync/models/attention.py:110-124` | attention 输出 residual add 后回到 `(b, c, f, h, w)`。 |
| self-attn 层定义 | `latentsync/models/attention.py:145-153` | `self.attn1 = Attention(...)`，没有传 `cross_attention_dim`，因此是 self-attn。 |
| self-attn forward | `latentsync/models/attention.py:177-182` | `attn1(norm_hidden_states) + hidden_states`。 |
| Attention QKV 定义 | `latentsync/models/attention.py:230-236` | `to_q`、`to_k`、`to_v`、`to_out`。 |
| self-attn / cross-attn 共同实现 | `latentsync/models/attention.py:250-280` | 如果 `encoder_hidden_states is None`，第 257 行回退为 `hidden_states`，即 self-attn。 |
| FlashAttention 调用 | `latentsync/models/attention.py:270-271` | 使用 `F.scaled_dot_product_attention(query, key, value, ...)`。 |

Self-attn 在 block 中的构造位置：

| block 类型 | 文件与行号 | 说明 |
| --- | --- | --- |
| `UNetMidBlock3DCrossAttn` | `latentsync/models/unet_blocks.py:205-216` | mid block 的 `Transformer3DModel`。 |
| `CrossAttnDownBlock3D` | `latentsync/models/unet_blocks.py:318-330` | down cross-attn block 的 `Transformer3DModel`。 |
| `CrossAttnUpBlock3D` | `latentsync/models/unet_blocks.py:576-588` | up cross-attn block 的 `Transformer3DModel`。 |

## 4. 蓝色部分：audio cross-attn

蓝色 cross-attn 由 `BasicTransformerBlock.attn2` 实现，只有 `add_audio_layer=True` 时才会创建。该 cross-attn 的 query 来自当前视觉 latent tokens，key/value 来自 `encoder_hidden_states`，也就是 Whisper audio embedding。

| 位置 | 文件与行号 | 作用 |
| --- | --- | --- |
| config 开启 audio layer | `configs/unet/stage2.yaml:53` | `add_audio_layer: true`。 |
| config 设置 audio dim | `configs/unet/stage2.yaml:57` | `cross_attention_dim: 384`。 |
| UNet 保存开关 | `latentsync/models/unet.py:89-90` | `self.use_motion_module`、`self.add_audio_layer`。 |
| add_audio_layer 传给 down blocks | `latentsync/models/unet.py:128-153` | down block 构造时传入 `add_audio_layer`。 |
| add_audio_layer 传给 mid block | `latentsync/models/unet.py:157-176` | mid block 构造时传入 `add_audio_layer`。 |
| add_audio_layer 传给 up blocks | `latentsync/models/unet.py:203-226` | up block 构造时传入 `add_audio_layer`。 |
| cross-attn 层定义 | `latentsync/models/attention.py:155-168` | 如果 `add_audio_layer`，创建 `norm2` 和 `attn2`，并传入 `cross_attention_dim`。 |
| cross-attn forward | `latentsync/models/attention.py:183-194` | 有 `attn2` 且有 `encoder_hidden_states` 时执行 audio cross-attn。 |
| audio embedding reshape | `latentsync/models/attention.py:183-185` | 若输入是 `(b, f, s, d)`，reshape 为 `(b*f, s, d)`，对齐 spatial attention 的 `(b*f, h*w, c)`。 |
| cross-attn QKV | `latentsync/models/attention.py:254-259` | query 来自视觉 hidden states，key/value 来自 audio encoder hidden states。 |
| cross-attn attention | `latentsync/models/attention.py:270-279` | scaled dot-product attention、linear projection、dropout。 |

Cross-attn 在各 block 中的执行位置：

| block 类型 | 文件与行号 | 执行顺序 |
| --- | --- | --- |
| `UNetMidBlock3DCrossAttn` | `latentsync/models/unet_blocks.py:249-254` | `attn(..., encoder_hidden_states=encoder_hidden_states)`。 |
| `CrossAttnDownBlock3D` | `latentsync/models/unet_blocks.py:377-382`、`latentsync/models/unet_blocks.py:393-394` | checkpoint 和普通路径都会执行 attention。 |
| `CrossAttnUpBlock3D` | `latentsync/models/unet_blocks.py:640-645`、`latentsync/models/unet_blocks.py:656-657` | checkpoint 和普通路径都会执行 attention。 |

补充：`BasicTransformerBlock` 在 cross-attn 后还会执行 FFN：

- FFN 定义：`latentsync/models/attention.py:170-172`
- FFN forward：`latentsync/models/attention.py:196-197`

## 5. 灰色部分：temporal layer / motion module

灰色 temporal layer 在代码里叫 `motion_module`，具体实现文件是 `latentsync/models/motion_module.py`。它的核心是 `VanillaTemporalModule -> TemporalTransformer3DModel -> TemporalTransformerBlock -> VersatileAttention`。

需要特别注意：`motion_module.py:3-5` 的注释写着作者最终版本“实际上不使用 motion module，结果较差，保留代码供未来使用”。但当前仓库的 `configs/unet/stage2.yaml:85-99` 又把 `use_motion_module: true` 打开，并且训练配置 `configs/unet/stage2.yaml:33-35` 只训练 `motion_modules.` 和 `attentions.`。因此排查时要以实际使用的 config / checkpoint 为准。

### 5.1 temporal layer 的开启与插入位置

| 位置 | 文件与行号 | 作用 |
| --- | --- | --- |
| config trainable modules | `configs/unet/stage2.yaml:33-35` | stage2 训练 `motion_modules.` 和 `attentions.`。 |
| config 开启 motion module | `configs/unet/stage2.yaml:85-99` | `use_motion_module: true`，`motion_module_type: Vanilla`，attention block 为两个 `Temporal_Self`。 |
| UNet 参数定义 | `latentsync/models/unet.py:77-83` | motion module 相关参数。 |
| down blocks 启用规则 | `latentsync/models/unet.py:147-152` | down block 在指定 resolution 且非 decoder-only 时插入 motion module。 |
| mid block 启用规则 | `latentsync/models/unet.py:172-175` | 只有 `motion_module_mid_block=True` 才在 mid block 插入；stage2 默认是 false。 |
| up blocks 启用规则 | `latentsync/models/unet.py:222-225` | up block 在指定 resolution 插入 motion module。 |

Temporal layer 在各 block 中的构造位置：

| block 类型 | 文件与行号 | 说明 |
| --- | --- | --- |
| `UNetMidBlock3DCrossAttn` | `latentsync/models/unet_blocks.py:218-226` | 如果 `use_motion_module`，append `get_motion_module(...)`，但 stage2 默认 mid 关闭。 |
| `CrossAttnDownBlock3D` | `latentsync/models/unet_blocks.py:332-340` | 每层 resnet/attention 后配置一个 motion module 或 None。 |
| `DownBlock3D` | `latentsync/models/unet_blocks.py:452-460` | 无 cross-attn block 也可以插入 motion module。 |
| `CrossAttnUpBlock3D` | `latentsync/models/unet_blocks.py:590-598` | decoder cross-attn block 插入 motion module。 |
| `UpBlock3D` | `latentsync/models/unet_blocks.py:713-720` | decoder 非 cross-attn block 插入 motion module。 |

Temporal layer 在 forward 中的执行位置：

| block 类型 | 文件与行号 | 执行顺序 |
| --- | --- | --- |
| `UNetMidBlock3DCrossAttn` | `latentsync/models/unet_blocks.py:256-258` | attention 后，如果存在 motion module，则执行 temporal，再执行 resnet。 |
| `CrossAttnDownBlock3D` | `latentsync/models/unet_blocks.py:384-391`、`latentsync/models/unet_blocks.py:396-397` | attention 后执行 temporal。 |
| `DownBlock3D` | `latentsync/models/unet_blocks.py:494-501`、`latentsync/models/unet_blocks.py:505-506` | resnet 后执行 temporal。 |
| `CrossAttnUpBlock3D` | `latentsync/models/unet_blocks.py:647-654`、`latentsync/models/unet_blocks.py:659-660` | attention 后执行 temporal。 |
| `UpBlock3D` | `latentsync/models/unet_blocks.py:759-766`、`latentsync/models/unet_blocks.py:770-771` | resnet 后执行 temporal。 |

### 5.2 temporal layer 的内部实现

| 位置 | 文件与行号 | 作用 |
| --- | --- | --- |
| motion module factory | `latentsync/models/motion_module.py:29-36` | `motion_module_type == "Vanilla"` 时返回 `VanillaTemporalModule`。 |
| `VanillaTemporalModule` | `latentsync/models/motion_module.py:39-73` | 包装 `TemporalTransformer3DModel`，可 zero-init `proj_out`。 |
| `TemporalTransformer3DModel` 定义 | `latentsync/models/motion_module.py:76-125` | temporal transformer 的 norm、projection、block list、output projection。 |
| temporal forward reshape | `latentsync/models/motion_module.py:126-151` | 先把 5D tensor reshape 到 spatial token 序列，执行 temporal transformer 后再 reshape 回 5D。 |
| `TemporalTransformerBlock` | `latentsync/models/motion_module.py:154-218` | 由多个 `VersatileAttention` 和 FFN 组成。 |
| positional encoding | `latentsync/models/motion_module.py:221-234` | 给 temporal sequence 加正弦位置编码。 |
| `VersatileAttention` 定义 | `latentsync/models/motion_module.py:237-260` | 继承普通 `Attention`，限制 `attention_mode == "Temporal"`。 |
| temporal attention reshape | `latentsync/models/motion_module.py:262-276` | `(b*f, s, c)` -> `(b*s, f, c)`，即每个空间位置沿时间帧做 attention。 |
| temporal QKV 和 attention | `latentsync/models/motion_module.py:280-309` | 与普通 attention 类似，执行 QKV、scaled dot-product attention、projection、dropout。 |
| reshape 回 spatial tokens | `latentsync/models/motion_module.py:310-311` | `(b*s, f, c)` -> `(b*f, s, c)`。 |

temporal layer 的关键语义是：spatial self-attn / cross-attn 在每帧内部的 `(h*w)` token 上做，而 temporal layer 将同一个空间位置跨帧组织成长度为 `f` 的序列，在时间维度做 attention。

## 6. 三段结构在 CrossAttn block 中的精确执行顺序

以 `CrossAttnDownBlock3D` 普通 forward 为例：

1. `latentsync/models/unet_blocks.py:393`：`hidden_states = resnet(hidden_states, temb)`，对应白色 conv / ResNet。
2. `latentsync/models/unet_blocks.py:394`：`hidden_states = attn(...).sample`，进入 `Transformer3DModel`。
3. `latentsync/models/attention.py:177-182`：`attn1` spatial self-attn，对应白色 self-attn。
4. `latentsync/models/attention.py:183-194`：`attn2` audio cross-attn，对应蓝色 cross-attn。
5. `latentsync/models/attention.py:196-197`：FFN。
6. `latentsync/models/unet_blocks.py:396-397`：如果存在 `motion_module`，执行 temporal layer，对应灰色 temporal layer。

`CrossAttnUpBlock3D` 的普通 forward 顺序相同，只是先在 `latentsync/models/unet_blocks.py:621-624` 取出 skip connection 并 `torch.cat`。

## 7. 排查时最建议看的最小文件集

| 文件 | 建议重点 |
| --- | --- |
| `latentsync/models/unet.py` | UNet 总体搭建、down/mid/up 的启用规则、forward 主流程。 |
| `latentsync/models/unet_blocks.py` | 每类 block 中 `resnet -> attention -> motion_module` 的构造和执行顺序。 |
| `latentsync/models/resnet.py` | 白色 conv / ResNet / upsample / downsample 的具体实现。 |
| `latentsync/models/attention.py` | 白色 self-attn 和蓝色 audio cross-attn 的具体实现。 |
| `latentsync/models/motion_module.py` | 灰色 temporal layer 的具体实现。 |
| `configs/unet/stage2.yaml` | 当前 stage2 是否启用 audio layer、motion module，以及训练哪些模块。 |
| `scripts/train_unet.py` | 训练时 UNet 输入、audio embedding、loss 前后的调用链。 |
| `latentsync/pipelines/lipsync_pipeline.py` | 推理时 UNet 输入、audio embedding、denoise loop 的调用链。 |

