# 基于深度学习的书法字体识别
## 设计说明
- 中华文化源远流长，其中汉字书法字体种类就十分之多，但在很多情况下未经系统学习的人们无法简单的识别其中的汉字，本设计应用深度学习对汉字书法字体识别进行设计，使其能够以较高的准确度识别出目的书法字体。
- 其中数据包括100个汉字书法单字，汉字类别有碑帖，手写书法，古汉字等等。图片全部为单通道灰度jpg，宽高不定。
- 模型最后测试集精确度达97.53%，数据来源参考自[TinyMind竞赛](https://www.tinymind.cn/competitions/41#overview).
### 数据集
- 第一部分每个汉字100张图片共第一部分每汉字100张图片共10000张图片，有标签。
- 第二部分测试数据每汉字50张以上图片（单字图片数不固定）共16346张图片，无标签，需上传至题目网站进行检测。
- 训练集：链接：https://pan.baidu.com/s/1ASOns2qH7D80JqZldxvo9A 提取码：d3za
- 测试集：链接：https://pan.baidu.com/s/1fxbO6e_gEJC9gNhaTPcrKg 提取码：emqi 
### 设计划分
#### 神经网络训练
1. [程序思路](train/README.md)
2. [程序源码](train/train.py)
#### 简单前端界面
1. [程序思路](interface/README.md)
2. [程序源码](interface/interface.py)