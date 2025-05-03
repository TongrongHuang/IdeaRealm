<div align="center">
    <img src="./images/logo-1.png" width="200" height="200">
    <!-- <div>&nbsp;</div> -->
    <div style="height: 5px;"></div>
    <h1>筑意绘界 IdeaRealm</h1>
</div>

## 简介
筑意绘界(IdeaRealm)是一个基于 CogView4 与 CogVideoX 1.5 的室内设计项目

“Idea”代表创意和想法，“Realm”则代表一个独特的领域或世界。

## 项目架构图
![Architecture diagram](./images/Architecture-diagram.Png)

## 项目安装说明
### 1、安装相关依赖
1. `conda create -n IdeaReam_env python=3.10`
2. 进入项目路径`path/IdeaRealm`，然后执行`pip install -r requirements.txt`安装项目所需依赖

### 2、下载 筑意绘界 所需模型
1. 如果是Linux系统，执行`export HF_ENDPOINT=https://hf-mirror.com`，如果是Windows系统，执行`$env:HF_ENDPOINT = "https://hf-mirror.com"`
2. 下载 CogView4 模型，在终端执行：`huggingface-cli download --resume-download THUDM/CogView4-6B --local-dir ./WebUI/models`
3. 下载 CogVideoX 1.5 模型，在终端执行：`huggingface-cli download --resume-download THUDM/CogVideoX1.5-5B --local-dir ./WebUI/models`
4. 下载 CogVideoX 1.5 I2V 模型，在终端执行：`huggingface-cli download --resume-download THUDM/CogVideoX1.5-5B-I2V --local-dir ./WebUI/models`
5. 下载 LoRA 模型，在终端执行：`huggingface-cli download --resume-download TongrongHuang/IdeaRealm_CogView4_LoRAs --local-dir ./WebUI/models`
6. （可选）下载 GLM4-9B 模型作为 CogView4 的 Encoder ：`huggingface-cli download --resume-download THUDM/glm-4v-9b --local-dir ./WebUI/models`，并且在`Gradio_UI.py`的61-70行代码中取消被注释的部分，把下载好的GLM路径写到第35行的`GLM_path`中。

### 3、运行WebUI
1. 在终端执行命令：`cd ./WebUI`
2. 接着在终端执行命令：`python Gradio_UI.py`

### 4、运行建议
显卡最好是N卡，并且至少拥有显存30G以上

## 项目完成情况
- [x] 实现文本生成室内精装实景图片
- [ ] 实现毛坯房图片生成室内精装实景图片
- [x] 实现文本生成室内精装实景视频
- [x] 实现室内实景图片生成室内实景视频
- [x] 成功部署
