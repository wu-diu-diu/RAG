---
title: dezerobug记录
date: 2024-07-23 21:09:48
tags: Dezero
---
# Dezero Bug 记录
在步骤52的时候，添加cupy库，引起一系列bug。
1.`UserWarning: Failed to auto-detect cl.exe path: <class 'distutils.errors.DistutilsPlatformError'>: No Microsoft Visual C++ version found`
即cupy库需要使用编译器构建C/C++扩展模块。在 Windows 系统上，通常使用 Microsoft Visual C++ 编译器（cl.exe）。如果你的系统上没有安装适当版本的 Visual C++，或者 Python 无法找到它的路径，就会出现这种错误。
### 解决方法1：
安装 VS 2022 时候会顺便下载C++编译器，，并将编译器的位置添加至环境变量，这样即可解决。[解决链接](https://blog.csdn.net/qq_40811682/article/details/118033631)。同时应注意编译器版本和cuda的兼容问题，第一次下载的编译器版本过新，后又下载了一个旧点的版本。同样是上面这个博主的文章[链接](https://blog.csdn.net/qq_40811682/article/details/118033177?spm=1001.2014.3001.5501)
### 解决方法2：
安装cupy的预编译版本：`pip install cupy-cuda11x`，11x是因为我电脑安装的cuda是11.7版本的。[链接](https://blog.csdn.net/weixin_45792450/article/details/138448774)。

**上述方法我两个都试了很多次，但最终还是报错，VS版本不支持之类的。重启电脑后错误消失，所以我也不知道到底哪个方法起效了哈哈**

## 过程中遇到问题
**1.Anaconda 崩溃**：无法创建环境，一直出错误报告。
### 解决方法
重装了一遍，想保留环境的话，将envs文件夹内的环境文件压缩保存，待新装的anaconda装好之后，再解压到envs文件夹中即可。

**2.pycharm无法切换环境**： 新建了环境，无法在pycharm中添加
### 解决方法
不选择添加conda环境，直接添加系统解释器[链接](https://blog.csdn.net/qq_43800449/article/details/134305402)

**3.conda创建新环境在C盘**：新建一个环境在C盘，导致我下载新的包时候找不到环境
### 解决方法
修改.condarc文件[链接](https://blog.csdn.net/QH2107/article/details/126246310)

**4.系统环境变量Path丢失**：不知道怎么环境变量里的Path路径没了，安装conda，cuda，运行hexo都需要添加环境变量。
### 解决方法
故新建了一个环境变量Path，参照网上的图片添加了相应的路径，也不知道对不对。**后续出现找不到命令的错误**，多半是因为该命令的exe文件所在的路径没有添加至环境变量当中。

**5.module 'cupy' has no attribute 'scatter_add'**
上面的问题都解决后，出现了这个小问题，原因是cupy版本过新，已经没有这个方法了

### 解决方法
在发生错误的位置，import cupyx，再编写cupyx.scatter_add是正确的，即cupyx这个包保留了这个方法。

### 总结
折腾了一天，终于可以用cupy运行dezero的程序了，也就是步骤52的程序终于成功运行。大功告成！！！
