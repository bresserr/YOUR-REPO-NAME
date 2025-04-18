---
title: FramePack图像到视频生成
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
license: mit
---

# FramePack

FramePack是一个图像到视频生成工具，利用扩散模型将静态图像转换为动态视频。

## 特点

- 使用单张图片生成流畅的动作视频
- 基于HunyuanVideo和FramePack架构
- 支持低显存GPU（最低6GB）运行
- 可以生成最长120秒的视频
- 使用TeaCache技术加速生成过程

## 使用方法

1. 上传一张人物图像
2. 输入描述所需动作的提示词
3. 设置所需视频长度（秒）
4. 点击"开始生成"按钮
5. 等待视频生成（生成过程是渐进式的，会不断扩展视频长度）

## 示例提示词

- "The girl dances gracefully, with clear movements, full of charm."
- "A character doing some simple body movements."
- "The man dances energetically, leaping mid-air with fluid arm swings and quick footwork."

## 注意事项

- 视频生成是倒序进行的，结束动作将先于开始动作生成
- 如果需要高质量结果，建议关闭TeaCache选项
- 如果遇到内存不足错误，可以增加"GPU推理保留内存"的值

## 技术细节

此应用基于[FramePack](https://github.com/lllyasviel/FramePack)项目，使用了Hunyuan Video模型和FramePack技术进行视频生成。该技术可以将输入上下文压缩为固定长度，使生成工作量与视频长度无关，从而在笔记本电脑GPU上也能处理大量帧。

---

原项目链接：[FramePack GitHub](https://github.com/lllyasviel/FramePack)
