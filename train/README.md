## 神经网络训练程序说明
### 实验框架要求
 - Ubuntu 16.04 
 - TensorFlow-gpu
 - CUDA10
 - Keras
 - Python 3.x
### 实验框架搭建
#### 安装Ubuntu 16.04
直接在Ubuntu官网下载Ubuntu桌面版，然后将其烧写进U盘，再进行安装即可。
#### 安装配置TensorFlow-gpu版本
首先需要安装NVIDIA显卡的软件程序包，直接执行如下命令:
```
# Add NVIDIA package repositories
# Add HTTPS support for apt-key
$ sudo apt-get install gnupg-curl
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
$ sudo apt-get update
$ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
$ sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
$ sudo apt-get update
```
然后是安装NVIDIA驱动，同时创建文件目录/usr/lib/nvidia。
```
$ sudo mkdir /usr/lib/nvidia
$ sudo apt-get install --no-install-recommends nvidia-410$ sudo apt-get update
```
此时先使用指令nvidia-smi进行检查，看是否安装成功，然后再进行开发运行库的安装。
```
$ sudo apt-get install --no-install-recommends \
        cuda-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0  \
        libcudnn7-dev=7.4.1.5-1+cuda10.0

```
最后安装TensorRT程序及CUDA10.0。
```
$ sudo apt-get update && \
            sudo apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
            && sudo apt-get update \
            && sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0

```
至此，就配置好了Ubuntu 16.04+CUDA10+TensorFlow-gpu的生产环境了。
#### 安装Keras库
Keras库可以直接使用Python的pip指令安装。首先我们进行pip安装：
```
$ sudo apt-get install python-pip python-dev
```
然后安装Keras库文件：
```
$ sudo pip install -U --pre pip setuptools wheel
$ sudo pip install -U --pre numpy scipy matplotlib scikit-learn scikit-image
$ sudo pip install -U --pre tensorflow-gpu
$ sudo pip install -U --pre keras
```
这样，我们就基本完成了实验框架的搭建（个别图片处理的库则自行使用pip安装即可）。
### 程序设计思路简述
训练程序中，我们搭建了三种神经网络的框架，分别是ResNet、AlexNet和一个简单测试网络框架，最终采用的是AlexNet网络。

采用AlexNet的原因有很多，其中主要的就是实验时采用的计算机计算能力较为一般，而AlexNet较为简单，训练速度较快，而使用ResNet进行调试则需要付出较大的时间开销，如果有计算能力强的计算机可以尝试一下。

设计的AlexNet网络是完全参考了AlexNet的原型，除了第一层的参数有些不同，后面的参数几乎相同。同时，为了避免过拟合，我们在网络中添加了Dropout正则化，并且对图像数据进行了扩充；为了提高学习效率，我们使用了学习率衰减。总的来说，整个模型还有优化的空间，进行更加细致的超参数调试，应该能够将正确率优化到99%以上（当前97.53%）。