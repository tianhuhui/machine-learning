# TensorFlow简介
TensorFlow官网：https://www.tensorflow.org/
中文社区：http://www.tensorfly.cn/

## 关于TensorFlow
TensorFlow? 是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。它灵活的架构让你可以在多种平台上展开计算，例如台式计算机中的一个或多个CPU（或GPU），服务器，移动设备等等。TensorFlow 最初由Google大脑小组（隶属于Google机器智能研究机构）的研究员和工程师们开发出来，用于机器学习和深度神经网络方面的研究，但这个系统的通用性使其也可广泛用于其他计算领域。
![TensorFlow](https://leanote.com/api/file/getImage?fileId=5986bd1fab6441463e000aa2)

## 什么是数据流图（Data Flow Graph）?
数据流图用“结点”（nodes）和“线”(edges)的有向图来描述数学计算。“节点”一般用来表示施加的数学操作，但也可以表示数据输入（feed in）的起点/输出（push out）的终点，或者是读取/写入持久变量（persistent variable）的终点。“线”表示“节点”之间的输入/输出关系。这些数据“线”可以输运“size可动态调整”的多维数据数组，即“张量”（tensor）。张量从图中流过的直观图像是这个工具取名为“Tensorflow”的原因。一旦输入端的所有张量准备好，节点将被分配到各种计算设备完成异步并行地执行运算。

## TensorFlow的特征
- 高度的灵活性
TensorFlow 不是一个严格的“神经网络”库。只要你可以将你的计算表示为一个数据流图，你就可以使用Tensorflow。你来构建图，描写驱动计算的内部循环。我们提供了有用的工具来帮助你组装“子图”（常用于神经网络），当然用户也可以自己在Tensorflow基础上写自己的“上层库”。定义顺手好用的新复合操作和写一个python函数一样容易，而且也不用担心性能损耗。当然万一你发现找不到想要的底层数据操作，你也可以自己写一点c++代码来丰富底层的操作。

- 真正的可移植性（Portability）
Tensorflow 在CPU和GPU上运行，比如说可以运行在台式机、服务器、手机移动设备等等。想要在没有特殊硬件的前提下，在你的笔记本上跑一下机器学习的新想法？Tensorflow可以办到这点。准备将你的训练模型在多个CPU上规模化运算，又不想修改代码？Tensorflow可以办到这点。想要将你的训练好的模型作为产品的一部分用到手机app里？Tensorflow可以办到这点。你改变主意了，想要将你的模型作为云端服务运行在自己的服务器上，或者运行在Docker容器里？Tensorfow也能办到。Tensorflow就是这么拽 :)

- 将科研和产品联系在一起
过去如果要将科研中的机器学习想法用到产品中，需要大量的代码重写工作。那样的日子一去不复返了！在Google，科学家用Tensorflow尝试新的算法，产品团队则用Tensorflow来训练和使用计算模型，并直接提供给在线用户。使用Tensorflow可以让应用型研究者将想法迅速运用到产品中，也可以让学术性研究者更直接地彼此分享代码，从而提高科研产出率。

- 自动求微分
基于梯度的机器学习算法会受益于Tensorflow自动求微分的能力。作为Tensorflow用户，你只需要定义预测模型的结构，将这个结构和目标函数（objective function）结合在一起，并添加数据，Tensorflow将自动为你计算相关的微分导数。计算某个变量相对于其他变量的导数仅仅是通过扩展你的图来完成的，所以你能一直清楚看到究竟在发生什么。

- 多语言支持
Tensorflow 有一个合理的c++使用界面，也有一个易用的python使用界面来构建和执行你的graphs。你可以直接写python/c++程序，也可以用交互式的ipython界面来用Tensorflow尝试些想法，它可以帮你将笔记、代码、可视化等有条理地归置好。当然这仅仅是个起点——我们希望能鼓励你创造自己最喜欢的语言界面，比如Go，Java，Lua，Javascript，或者是R。

- 性能最优化
比如说你又一个32个CPU内核、4个GPU显卡的工作站，想要将你工作站的计算潜能全发挥出来？由于Tensorflow 给予了线程、队列、异步操作等以最佳的支持，Tensorflow 让你可以将你手边硬件的计算潜能全部发挥出来。你可以自由地将Tensorflow图中的计算元素分配到不同设备上，Tensorflow可以帮你管理好这些不同副本。

---------------------------------
# TensorFlow的安装
参照TensorFlow的官网安装说明（https://www.tensorflow.org/install/），TensorFlow提供了多种安装方式的教程，综合分为两类：

- 一是直接利用官方或者第三方编译好的release版本进行安装
- 二是利用官方的源代码编译生成包后安装

我自己的电脑目前安装的是WIN10系统，因此参照的是Installing TensorFlow on Windows说明。

## 选择TensorFlow版本类型
TensorFlow版本包括2种：只支持CPU的版本、支持GPU的版本。我的笔记本没有安装GPU，因此只能装支持CPU版本的。

## 如何安装TensorFlow
支持通过pip或者Anaconda来安装TensorFlow。
我没有安装Anaconda，用的是pip。

## 通过pip安装
TensorFlow在Windows系统上只支持3.5版本及之后版本的Python。我的WIN10系统上安装的是3.6版本的Python，OK。

开始时，我是用的如下命令安装的：
pip install --upgrade --ignore-installed tensorflow

但是总是会在中途失败，查看报错信息，出现
![title](https://leanote.com/api/file/getImage?fileId=5986c41cab644143e4000b51)
我估计是因为网速太慢的缘故，因此我直接上https://pypi.python.org/pypi/tensorflow下载安装包tensorflow-1.3.0rc2-cp36-cp36m-win_amd64.whl。

下载完成后，本地pip安装：

    pip install ./tensorflow-1.3.0rc2-cp36-cp36m-win_amd64.whl

查看安装进度，发现又有安装失败的错误，网络下载超时了
![title](https://leanote.com/api/file/getImage?fileId=5986c4c7ab6441463e000b5b)

重新再试一次，OK。
![title](https://leanote.com/api/file/getImage?fileId=5986c4eeab6441463e000b67)

## 验证安装是否成功
在Python Shell中测试如下命令：

    >>> import tensorflow as tf
    >>> hello = tf.constant('Hello, TensorFlow!')
    >>> sess = tf.Session()
    >>> print(sess.run(hello))
    
输出：

    Hello, TensorFlow!
    
不过我的终端输出是这样的：

    b'Hello, TensorFlow!'

为何会字符串前面有个b呢？待续……

-----------------------------
# 参考资料
TensorFlow官网：https://www.tensorflow.org/
中文社区：http://www.tensorfly.cn/
http://www.jianshu.com/p/4b0992253626
https://zhuanlan.zhihu.com/p/24055668