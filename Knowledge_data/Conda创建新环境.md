---
title: Conda创建新环境及镜像源设置
date: 2024-05-08 15:43:39
tags: machine learning
---

## Conda创建新环境步骤

### 第一步：

conda创建一个新的环境：`conda create -n 环境名 python=x.x`，注意，python版本要和安装的torch版本相**对应**，可在[该网址](https://blog.csdn.net/WOSHIRENXIN/article/details/127415609)查询其对应关系
### 第二步：

安装torch包：去[pytorch官网](https://pytorch.org/get-started/previous-versions/) 下载**GPU**版本的torch，torchvision包，即选择带有**+cuxxx**的版本
### 第三步：查询并安装对应版本的cuda和cudnn

使用`conda search cudatoolkit`和`conda search cudnn`来确定可安装的版本

### 第四步：检查
命令行输入python进入编辑器，依次输入：`import torch`, `print(torch.__version__)`, `print(torch.cuda.is_available())`,结果为Ture，则环境配置成功。cuda和cudnn版本对应可参考[网址](https://blog.csdn.net/matrix273/article/details/103534991)

**注意**：安装包时若出现一大行warning，检查校园网是否连接，梯子是否关闭，梯子光直连还不行，要退出才行。

### Conda安装和pip安装区别
conda安装时，会检查各个包之间的依赖关系，版本是否对应，即conda安装后会保证包能够正常运行，而pip安装则不行，而且有的包用pip安装不了用conda却可以。所以建议**一直使用一种方式安装**。

### 安装镜像源设置
清华大学镜像源：
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/`
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/`
显示已添加的源：
`conda config --show channels`

添加镜像源：`conda config --append channels conda-forge
`

