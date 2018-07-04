## scikit-learn介绍
scikit-learn是Python的一个开源机器学习模块，它建立在NumPy、SciPy和matplotlib模块之上，详细介绍可以参照scikit-learn官网：http://scikit-learn.org/stable/

-------------------------------------
## 安装准备工作
- Python
- Numpy

NumPy是Python的一种开源的数值计算扩展工具，可用来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure）结构要高效的多（该结构也可以用来表示矩阵）。

- Scipy

SciPy是一款方便的、易于使用的、专门为科学和工程设计的Python工具包，它包括统计、优化、整合、线性代数模块、傅里叶变换、信号和图像处理、常微分方程求解器等。

- matplotlib

matplotlib是python最著名的绘图库，它提供了一整套和MATLAB详细的命令API，十分适合交互式绘图，而且可以方便地将它作为绘图控件，嵌入到GUI应用程序中。

### 下载地址
Python：https://www.python.org/downloads/
NumPy：https://pypi.python.org/pypi/numpy
（or http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy，链接了 mkl 与未链接 mkl 的numpy在性能上会有显著差异，你可以前往上述网址安装链接了 mkl 库的 numpy）
SciPy：http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
matplotlib：http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib
sciki-learn：http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn

我自己安装的版本如下：
- Python：3.6.2（64bit）
- NumPy：numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl
- SciPy：scipy-0.19.1-cp36-cp36m-win_amd64.whl
- matplotlib：matplotlib-2.0.2-cp36-cp36m-win_amd64.whl
- scikit-learn：scikit_learn-0.18.2-cp36-cp36m-win_amd64.whl

文件名组成：
**库名-库版本号-python版本-平台**

例如：
scikit_learn-0.18.2-cp36-cp36m-win_amd64.whl
代表的是：
scikit-learn版本号0.18.2
cp36代表python3.6
win_amd64代表Windows 64位系统

--------------
## 安装步骤
首先安装Python，记得勾选加入到环境变量。
然后依次安装NumPy、SciPy和matplotlib：

- 安装NumPy：
pip install ".\numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl"
![title](https://leanote.com/api/file/getImage?fileId=597345b9ab644127c700170c)

安装成功！
（如果出现XXX is not a supported wheel on this platform错误提示，则仔细检查下python版本和whl版本是否一致）。

- 安装SciPy：
pip install .\scipy-0.19.1-cp36-cp36m-win_amd64.whl
![title](https://leanote.com/api/file/getImage?fileId=597346e8ab644127c7001726)

- 安装matplotlib
pip install .\matplotlib-2.0.2-cp36-cp36m-win_amd64.whl
![title](https://leanote.com/api/file/getImage?fileId=59734771ab644127c7001732)

- 安装scikit-learn
pip install .\scikit_learn-0.18.2-cp36-cp36m-win_amd64.whl
![title](https://leanote.com/api/file/getImage?fileId=597347c9ab644125a100168b)

-----------------------
## 测试scikit-learn
我们根据scikit-learn的入门指导（http://scikit-learn.org/stable/tutorial/basic/tutorial.html），测试下是否安装成功，功能是否可用。

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> digits = datasets.load_digits()
    >>> print(digits.data)  
    [[  0.   0.   5. ...,   0.   0.   0.]
     [  0.   0.   0. ...,  10.   0.   0.]
     [  0.   0.   0. ...,  16.   9.   0.]
     ...,
     [  0.   0.   1. ...,   6.   0.   0.]
     [  0.   0.   2. ...,  12.   0.   0.]
     [  0.   0.  10. ...,  12.   1.   0.]]
    >>> digits.target
    array([0, 1, 2, ..., 8, 9, 8])