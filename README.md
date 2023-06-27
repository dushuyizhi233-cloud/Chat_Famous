# Chat_Famous

## 项目介绍
这是一个基于Langchain和Stable Diffusion的视频生成程序，该视频包括基于用户查询AI生成的音频和基于查询的AI生成图像。
目的是实现和一位知名人物实现对话，在本项目中我们以马斯克为例。
输入的内容包含三部分，一是关于马斯克的语料文本，二是想问马斯克的问题，三是关于马斯克视觉生成的提示词。

核心代码在main.py文件中

## 目前实现效果如下（示例截图）
![image](https://github.com/dushuyizhi233-cloud/Chat_Famous/assets/61072937/c05aced7-4fec-4284-bb7d-6a9d599ab603)

目前使用的是河南话语音库

https://github.com/dushuyizhi233-cloud/Chat_Famous/assets/61072937/2ba8d3ff-6214-4eac-843f-4b2793ec01d0


## 说明
1.需要本地部署Stable Diffusion Web-UI

2.人脸识别构建3D动态效果部分通过调用Github开源项目SadTalker实现。

3.为使StableDiffusion生成更具马斯克特征的图片，我们通过搜集的20张马斯克的照片训练了一个LoRA（Low-Rank Adaptation of Large Language Models ）模型，该训练方法的原理是冻结预训练模型参数，在每个Transformer块插入可训练层，不需要完整调整 UNet 模型的全部参数。训练结果只保留新增的网络层，模型体积小。
LoRA 是一种轻量级的微调方法，通过少量的图片训练出一个小模型，然后和基础模型结合使用，并通过插层的方式影响模型结果。LoRA 的优势在于生成的模型较小，训练速度快，但推理需要同时使用 LoRA 模型和基础模型。LoRA 模型虽然会向原有模型中插入新的网络层，但最终效果还是依赖基础模型。

# 未来完善思路
1.语音模型的训练

2.连续对话的实现

3.更丰富的训练文本

2.更高质量的embedding和对中文更友好的LLM模型
