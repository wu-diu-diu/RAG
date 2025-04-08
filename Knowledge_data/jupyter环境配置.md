---
title: jupyter环境配置
date: 2024-05-28 08:18:57
tags: machine learning
---
## 添加所有环境至jupyter
**代码如下：**
1.`conda activate my-conda-env`
2.`conda install ipykernel`
3.`conda deactivate`
4.`conda activate base`
5.`conda install nb_conda_kernels`
6.`jupyter notebook`

**这样打开jupyter后，就可以随意切换环境了**

[参考链接](https://blog.csdn.net/u014264373/article/details/119390267)