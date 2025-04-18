# FramePack - 图像到视频生成

![FramePack封面图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-spaces/gradio-banner.png)

将静态图像转换为动态视频的人工智能应用。上传一张人物图像，添加动作描述，即可生成流畅的视频！

## 使用方法

1. 上传一张人物图像
2. 输入描述所需动作的提示词（如"The girl dances gracefully"）
3. 调整视频长度和其他可选参数
4. 点击"开始生成"按钮
5. 等待视频生成（过程是渐进式的，会不断扩展视频长度）

## 示例提示词

- "The girl dances gracefully, with clear movements, full of charm."
- "The man dances energetically, leaping mid-air with fluid arm swings and quick footwork."
- "A character doing some simple body movements."

## 技术特点

- 基于Hunyuan Video和FramePack架构
- 支持低显存GPU运行
- 可生成最长120秒的视频
- 使用TeaCache技术加速生成过程

## 注意事项

- 视频生成是倒序进行的，结束动作将先于开始动作生成
- 首次使用时需要下载模型（约30GB），请耐心等待
- 如果遇到内存不足错误，可以增加"GPU推理保留内存"的值

---

原项目: [FramePack GitHub](https://github.com/lllyasviel/FramePack) 