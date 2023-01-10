---
title: Ubuntu20.04安装深度学习环境及配置个人环境小记
date: 2021-10-30 12:22:01
tags: [Linux]
categories: [笔记]
---

Ubuntu20.04安装深度学习环境及配置个人环境小记

<!-- more -->

### 目标配置环境

系统为Ubuntu 20.04, 显卡为3090，结合其他服务器的环境，准备配置如下环境

```bash
nvidia-driver-460 # 英伟达驱动，最低推荐版本为这个
CUDA 11.2 (gcc 7) # 30系显卡应该换为cuda11版本以上，20系显卡可以用cuda10.2版本
cudnn 8.6.1

pytorch 1.10
torchvision 0.11.1
tensorflow 2.5.0
```

### 准备工作

刚装的系统，啥也没有，所以按照个人习惯先安装一些软件。为方便起见，后文非特殊说明，均用root用户完成安装。可以直接复制下文至sh文件中，更改权限以直接运行。（请保证软件源可正常连接）

```bash
# 第一步: 安装一些必备的软件
add-apt-repository ppa:graphics-drivers/ppa
apt update -y && apt upgrade -y
apt install -y git autossh python3-pip vim tmux gpustat tree ranger build-essential gcc-7 g++-7 make curl htop ipython3 zhs
# gcc与g++降级
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100

# 第二步：安装nvidia驱动
# 禁用nouveau
chmod 666 /etc/modprobe.d/blacklist.conf
# 增加相应配置, 若已追加则记得注释下面这一行命令
echo "
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist rivatv
blacklist nvidiafb
"
>> /etc/modprobe.d/blacklist.conf
# 更新配置
update-initramfs -u
apt install nvidia-driver-460 -y # 英伟达460驱动，按需更改
```

执行完成之后，重启服务器。使用`nvidia-smi`查看gpu是否正常，或使用安装好的`gpustat`

### cuda及cudnn安装

#### 下载

这里需要从官网下载安装包，cuda安装包可以直接通过命令下载。cudnn需要注册英伟达开发者账号，下载四个文件

- `cudnn-11.2-linux-x64-v8.1.0.77`
- `libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb `

- `libcudnn8-dev_8.1.0.77-1+cuda11.2_amd64.deb`
- `libcudnn8-samples_8.1.0.77-1+cuda11.2_amd64.deb `

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run && chmod 755 cuda_11.2.2_460.32.03_linux.run

./cuda_11.2.2_460.32.03_linux.run
```

下载完直接运行，安装过程中需要把Drivers这项取消，因为自带了显卡驱动。

#### 环境变量

在/etc/bash.bashrc或/etc/profile前加入如下两行。如果profile不行就换bashrc。一般是编辑`vim ~/.bashrc`下，但为保证所有用户都生效，所以需要在`/etc`目录下编辑

```bash
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

至此，cuda安装完成

```bash
# 检测是否安装成功
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make clean && make -j4 && ./deviceQuery
# 最后一行输出，Result = PASS表示成功
```

#### cudnn配置

```bash
tar -zxvf cudnn-11.2-linux-x64-v8.1.0.77.tgz 
cp cuda/include/cudnn.h /usr/local/cuda/include 
cp cuda/lib64/libcudnn* /usr/local/cuda/lib64  
chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*  

dpkg -i libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb      
dpkg -i libcudnn8-dev_8.1.0.77-1+cuda11.2_amd64.deb  
dpkg -i libcudnn8-samples_8.1.0.77-1+cuda11.2_amd64.deb 

# cudnn检测，更高版本的cudnn需要把cudnn.h换成cudnn_version.h
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
# 利用程序检查，可能需要重启
cd /usr/src/cudnn_samples_v7/conv_sample
make clean && make -j4 && ./conv_sample
# 最后一行输出，Test passed！表示成功
```

通过`ldconfig`可检查动态链接库是否链接正确。若有输出，可以使用`ln -sf`重新链接。

### Python库安装

```bash
pip3 install tensorflow-gpu==2.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install black tqdm flask sklearn
```

#### 检查是否安装成功

```python
import torch

print(torch.cuda.is_available()) 
print(torch.backends.cudnn.is_available()) 

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

import tensorflow as tf
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)   
tf.test.is_gpu_available() # 输出True
```

若tf未检测到GPU，可以看输出过程中的哪些动态库文件加载失败，手动配置一下。

至此，深度学习安装环境配置完成。

### 个人环境

```bash
# 创建用户
adduser xxx
# 输入配置的密码

# 增加root权限
chmod u+w /etc/sudoers
vim /etc/sudoers
​```
root    ALL=(ALL:ALL) ALL
# 增加一行
xxx    ALL=(ALL:ALL) ALL
​```
chmod u-w /etc/sudoers
```

#### ssh文件配置

方便起见直接把其他服务器的`.ssh`文件夹下的`id_rsa`，`id_rsa.pub`和`authorized_keys`复制过来。

```bash
# 复制完成之后，修改对应权限才能使用
chmod 755 ~/.ssh/  
chmod 600 ~/.ssh/id_rsa ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys
chmod 644 ~/.ssh/known_hosts
```

#### oh-my-zsh

```bash
# 下载并安装
wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | sh

# 安装插件
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# vim ~/.zshrc, 最后一行加入如下配置
plugins=(
        git
        zsh-autosuggestions
        zsh-syntax-highlighting
)
source $ZSH/custom/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
source $ZSH/custom/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh
source $ZSH/oh-my-zsh.sh
```

#### 配置vim

编辑`.vimrc`

```bash
set number

syntax on
colorscheme darkblue


syntax enable
filetype plugin on
let g:go_disable_autoinstall = 0
let g:go_highlight_functions = 1
let g:go_highlight_methods = 1
let g:go_highlight_structs = 1
let g:go_highlight_operators = 1
let g:go_highlight_build_constraints = 1
" " syntax on
"" filetype plugin indent on
let g:pymode_python = 'python3'
set number
set cin
set sw=8
set ts=8
set sm
colorscheme darkblue
set viminfo='10,\"100,:20,%,n~/.viminfo 

au BufReadPost * if line("'\"") > 0|if line("'\"") <= line("$")|exe("norm '\"")|else|exe "norm $"|endif|endif
```

#### 远程连接

```bash
autossh -gfCNR 远程端口:localhost:22（默认本地ssh端口） 公网用户名@公网IP
```

需要提前配好免密登录，远程端口在公网服务器上需要开放防火墙。

远程连接登录`ssh 服务器用户@公网IP -p 远程端口`

