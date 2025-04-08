---
title: GPT2环境配置
date: 2024-08-04 17:28:27
tags: 深度学习环境配置
---
### 环境配置记录
本地配置方法：去[pytorch官网](https://pytorch.org/get-started/previous-versions/)找到要下载的历代版本，比如本次配置版本为
	
	conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
直接使用这个命令即可下载pytoch相关包以及cuda包，此处cuda包的版本**不需要**非得和电脑上安装的cuda软件版本一致。最后在下载一个cudnn的包即可

### 速度太慢
直接使用上述命令的速度太慢，故采用本地下载方式。

1.去[清华镜像网站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/)找到对应python版本的pytorch，torchvision和torchaudio包的文件并下载，并使用下述命令下载到虚拟环境：

	conda install --use-local C:\Users\武丢丢\Downloads\文件名

2.使用conda search cuda --info命令，得到适配的cuda版本及下载网址，cudnn同理。得到cuda和cudnn的本地包之后，用步骤1命令下载到虚拟环境。

3.下载完进入虚拟环境后运行`import torch`会报错**shm.dll**找不到,这是在虚拟环境中再次运行pytorch官方命令：（**这次会快很多**）

	conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

即可解决错误

### 注意
若遇到conda无法创建新环境的问题，重启电脑多半可以解决