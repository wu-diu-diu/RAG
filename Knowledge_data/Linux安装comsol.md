---
title: Linux安装comsol
date: 2025-03-31 10:50:03
tags: Linux
---
# Linux安装comsol
在linux系统中安装comsol和在win中有所不同，接下来从下载压缩包，解压，创建文件夹，挂载，安装等步骤讲解如何在linux系统中安装一个comsol。
## 下载iso文件
首先到IT技术之家下载comsol的iso文件，版本是comsol6.3。[下载地址](https://www.ittel.cn/archives/6828.html)下载好后需要上传到远程linux服务器，可以使用xftp或者ssh，如果你是直接在现场操作的话则不需要上传了，拷贝一下就行。注意如果U盘插入到电脑中但却点不开的话，**可能需要先将U盘挂载到某一个文件夹下**，这样才能将U盘中的文件取出
## 解压rar文件
上面下载的压缩包是带密码的，建议使用unrar解压,`sudo apt install unrar`，cd到解压的文件夹，命令为：`rar x  filename.rar`
## 将iso文件挂载到指定目录
可以在/mnt目录下新建一个目录用于挂载，因为/mnt 文件夹是一个用于临时挂载文件系统的目录。它通常用于挂载外部存储设备（如 USB 驱动器、外部硬盘、CD-ROM 等）或虚拟文件系统（如 ISO 文件）。/mnt 文件夹在系统启动时通常为空，用户可以根据需要手动创建子目录并挂载文件系统。参考链接如下：
[linux下运行comsol,COMSOL5.3在Linux下的安装](https://blog.csdn.net/weixin_35651916/article/details/116623625)
[comsol 安装说明 --- 本文介绍通过登录远程linux 终端安装comsol的方法](https://blog.csdn.net/sowhatgavin/article/details/70666200)
- `sudo mkdir /mnt/comsol`
- `mv /home/user/Downloads/example.iso /mnt/comsol/`
- `sudo mount -o loop -t iso9660 /home/comsol/COMSOL62_dvd_Windows_Linux.iso /mnt/comsol`
- 运行`df -lh`查看挂载情况
- 挂载成功后进入挂载点文件夹，`cd /mnt/comsol`
- 执行`sudo ./setup`
- 按照提示完成安装即可
