# Chat_Famous
核心代码在main.py文件中

说明：人脸识别构建3D动态效果部分通过调用Github开源项目SadTalker实现。
此外，为使StableDiffusion生成更具马斯克特征的图片，我们通过搜集的20张马斯克的照片训练了一个LoRA（Low-Rank Adaptation of Large Language Models ）模型，该训练方法的原理是冻结预训练模型参数，在每个Transformer块插入可训练层，不需要完整调整 UNet 模型的全部参数。训练结果只保留新增的网络层，模型体积小。
LoRA 是一种轻量级的微调方法，通过少量的图片训练出一个小模型，然后和基础模型结合使用，并通过插层的方式影响模型结果。LoRA 的优势在于生成的模型较小，训练速度快，但推理需要同时使用 LoRA 模型和基础模型。LoRA 模型虽然会向原有模型中插入新的网络层，但最终效果还是依赖基础模型。
