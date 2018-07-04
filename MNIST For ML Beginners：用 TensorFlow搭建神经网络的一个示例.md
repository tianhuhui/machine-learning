官网链接：
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/

MNIST是一个简单的机器视觉数据集，它包括手写数字的图片，例如：
![title](https://leanote.com/api/file/getImage?fileId=59872c9eab6441463e0016a5)

数据集也包括了每个图片的标签（label），告诉我们是什么数字，例如上面几个图片的标签依次是5、0、4、1。

------------------------------
# MNIST数据集
数据集在网站http://yann.lecun.com/exdb/mnist/上，以下两行可以自动下载和导入数据集：

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

不过经过测试，下载的速度非常慢，可以直接从数据集网站上手动下载数据集，下载以下四个文件：
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)



MNIST数据集包含三部分：55000个数据用于训练，10000个数据用于测试，5000个数据用于验证。

如前所述，每个MNIST数据包含2部分：一个手写数字的图像和一个对应的标签，我们可以称图像“x”和标签“y”。

每一张图片是28*28像素的，我们可以把这个图片解释为一个大的数组：
![title](https://leanote.com/api/file/getImage?fileId=598731b4ab6441463e00171f)

我们可以把这个数组平铺成一个28*28=784维的向量，这样我们就可以把每张图片看成一个784维向量空间中的一个点。

这样的话，mnist.train.images就是一个tensor，其shape是[55000, 784]。
![title](https://leanote.com/api/file/getImage?fileId=598732ccab6441463e001738)

在本示例中，我们想把数据标签变成“one-hot向量”，所谓的one-hot向量就是一个向量，只有一个维度是1，其余均是0.在本例子中，第n个数字可以表示为一个在第n维是1的一个one-hot向量，例如数字3对应的就是[0,0,0,1,0,0,0,0,0,0]. 相应的，mnist.train.labels就是shape是[55000, 10]的一个tensor。
![title](https://leanote.com/api/file/getImage?fileId=598733e0ab644143e40015e2)

-----------------------------
# Softmax回归
这里我们定义一个最简单的单层全连接网络，计算公式为：y=Wx+b,然后利用softmax来计算预测概率，预测概率最大的对应预测的分类。

![title](https://leanote.com/api/file/getImage?fileId=59873456ab6441463e00177a)

![title](https://leanote.com/api/file/getImage?fileId=59873473ab644143e40015ea)

![title](https://leanote.com/api/file/getImage?fileId=59873482ab644143e40015ec)

---------------------------
# 代码实现
## 定义神经网络模型
首先import TensorFlow：

    import tensorflow as tf

输入x：

    x = tf.placeholder(tf.float32, [None, 784])

定位变量W和b：

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

定义我们的模型，只需要一行搞定：

    y = tf.nn.softmax(tf.matmul(x, W) + b)

## 训练模型
我们采用cross-entropy作为损失函数：
![title](https://leanote.com/api/file/getImage?fileId=598737f8ab6441463e0017d2)

其中，![title](https://leanote.com/api/file/getImage?fileId=59873e80ab6441463e00184c)是我们预测的分布，![title](https://leanote.com/api/file/getImage?fileId=59873eaaab644143e40016e1)是真实的分布。

定义真实输出值：

    y_ = tf.placeholder(tf.float32, [None, 10])

定义损失函数：

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

选择优化算法，这里选择的是梯度下降法：

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

定义会话，运行运算和初始化变量：

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

开始训练参数：

    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      
## 评估模型

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
最后准确率约为92%。

-------------------------------
# 完整代码如下：

    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

说明：经过测试，下载数据集的速度非常慢，可以直接从数据集网站上手动下载数据集，下载以下四个文件：
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
然后在上述程序所在的同一个目录下，新建文件夹MNIST_data，把以上四个文件放到文件夹里面。

程序运行结果：
![title](https://leanote.com/api/file/getImage?fileId=5991b7dfab644147bd002578)
结果表明，最后的模型准确率为91.72%。

官方GitHub代码地址：
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py

------------------------------------
# 参考资料
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/
http://yann.lecun.com/exdb/mnist/
http://www.jianshu.com/p/87581c7082ba


