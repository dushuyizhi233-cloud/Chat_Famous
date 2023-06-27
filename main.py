import subprocess
from gtts import gTTS
import requests
import subprocess  # 这个库允许我们运行命令行指令
from gtts import gTTS  # Google Text-to-Speech库，将文本转化为音频
import requests  # 发送http请求
from PIL import Image  # 用于处理图像
import io  # 用于处理流（stream）操作
import base64  # 用于处理base64编码
import os  # 用于处理操作系统相关的操作，比如环境变量
os.environ["OPENAI_API_KEY"] = ''  # 设置OpenAI的API key
import re  # 正则表达式库，用于处理字符串
import jieba as jb  # 结巴分词，用于处理中文文本
from langchain.embeddings.openai import OpenAIEmbeddings  # 用于处理文本嵌入
from langchain.vectorstores import Chroma  # 用于存储和查询向量
from langchain.text_splitter import TokenTextSplitter  # 用于将文本切割成小块
from langchain.document_loaders import DirectoryLoader  # 用于加载文档
from langchain.chains.question_answering import load_qa_chain  # 用于加载问答链
from langchain import OpenAI,VectorDBQA  # 用于处理问答


# 定义函数，用于生成图片
def generate_image(prompt_template):
    # 图片生成服务的url，需要在本地部署Stable diffusion web-ui
    url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
    # 数据参数
    data = {
        "prompt": prompt_template,  # 输入模板
        "sampler_name": "DPM++ 2M Karras",  # 采样器名
        "batch_size": 1,  # 批次大小
        "steps": 20,  # 步骤数
        "cfg_scale": 7,  # 配置尺度
        "width": 512,  # 图片宽度
        "height": 512,  # 图片高度
        "negative_prompt": "blurry,low quality,text,car",  # 负面提示，生成图片时避免这些特征
    }
    # 发送post请求，并获取响应
    response = requests.post(url=url, json=data)
    # 从响应中提取图像数据
    image_data = response.json()['images'][0]
    # 将图像数据转化为图像，并保存为new_image.jpg
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image.save('new_image.jpg')

# 定义函数，用于预处理中文文本
def preprocess_chinese_text(text):
    # 使用jieba分词
    seg_list = jb.cut(text)
    tokenized_text = " ".join(seg_list)
    # 定义停用词
    stop_words = set(["的", "了", "是", "我", "你", "他"])
    # 去除停用词
    tokenized_text = " ".join([word for word in tokenized_text.split() if word not in stop_words])
    # 去除标点符号
    tokenized_text = re.sub(r"[^\w\s]", "", tokenized_text)
    # 将所有字母转为小写
    tokenized_text = tokenized_text.lower()
    # 返回预处理后的文本
    return tokenized_text

# 定义函数，用于预处理文档
def preprocess_documents(files, base_path):
    # 对每一个文件进行处理
    for file in files:
        # 获取文件路径
        my_file = os.path.join(base_path, file)
        # 打开文件，读取数据
        with open(my_file, "r", encoding='utf-8') as f:
            data = f.read()
        # 预处理数据
        cut_data = preprocess_chinese_text(data)
        # 获取预处理后数据的保存路径
        cut_file = os.path.join(base_path, 'cut', f"cut_{file}")
        # 保存预处理后的数据
        with open(cut_file, 'w', encoding='utf-8') as f:
            f.write(cut_data)

# 定义函数，用于加载文档
def load_documents(base_path):
    # 初始化文档加载器
    loader = DirectoryLoader(os.path.join(base_path, 'cut'), glob='**/*.txt')
    # 加载文档
    documents = loader.load()
    # 初始化文本切割器
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
    # 切割文档
    split_docs = text_splitter.split_documents(documents)
    # 返回切割后的文档
    return split_docs

# 定义函数，用于搜索文档并获取答案
def search_documents(query_template, split_docs):
    # 初始化嵌入
    embeddings = OpenAIEmbeddings()
    # 从文档创建向量搜索器
    docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory="D:/vector_store")
    # 定义查询
    person='马斯克'
    query = '在2008年发生了什么'
    query_template = f'请以{person}的第一人称口吻回答以下问题\n{query}'
    # 搜索文档
    docs = docsearch.similarity_search(query, include_metadata=True)
    # 初始化问答链
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    # 获取答案
    answer = chain.run(input_documents=docs, question=query_template)
    # 返回答案
    return answer

# 定义函数，用于生成音频
def generate_audio(text, language):
    # 设置语音配置
    voice_config = {'lang': language}
    # 将文本转化为音频
    tts = gTTS(text=text, **voice_config)
    # 保存音频
    tts.save('output.mp3')

# 定义函数，用于生成视频
def generate_video(image_path, audio_path, output_path):
    # 获取音频时长
    duration_cmd = ['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")]
    audio_duration = float(subprocess.check_output(duration_cmd).strip())
    # 定义视频生成命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-loop', '1',
        '-i', image_path,
        '-i', audio_path,
        '-c:v', 'libx264',
        '-t', str(audio_duration),
        '-pix_fmt', 'yuv420p',
        '-shortest',
        output_path
    ]
    # 运行视频生成命令
    subprocess.run(ffmpeg_cmd, check=True)
    # 输出视频生成成功的信息
    print("Video created successfully at", output_path)

# 配置信息
person = 'alon musk'
prompt_template = f'1person,({person}),high quality,closeup'

# 生成图片
generate_image(prompt_template)
print('图片生成完毕！')

# 预处理文档
base_path = r"C:\Users\Dabao\Documents\python_projects\musk_content"
files = ['test2.txt']
preprocess_documents(files, base_path)

# 加载文档
documents = load_documents(base_path)
print(len(documents))

# 搜索并获取答案
query = f'{person}在2008年发生了什么事情'
query_template = f'请以{person}的第一人称口吻回答以下问题\n{query}'
answer = search_documents(query_template, documents)
print(answer)

# 生成音频
language = 'zh-cn'
generate_audio(answer, language)
print('音频生成完毕！')

# 合成视频
image_path = "./new_image.jpg"
audio_path = "./output.mp3"
output_path = "./output.mp4"
generate_video(image_path, audio_path, output_path)
print('视频生成完毕！')（文本）
