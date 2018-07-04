�������ӣ�
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/

MNIST��һ���򵥵Ļ����Ӿ����ݼ�����������д���ֵ�ͼƬ�����磺
![title](https://leanote.com/api/file/getImage?fileId=59872c9eab6441463e0016a5)

���ݼ�Ҳ������ÿ��ͼƬ�ı�ǩ��label��������������ʲô���֣��������漸��ͼƬ�ı�ǩ������5��0��4��1��

------------------------------
# MNIST���ݼ�
���ݼ�����վhttp://yann.lecun.com/exdb/mnist/�ϣ��������п����Զ����غ͵������ݼ���

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

�����������ԣ����ص��ٶȷǳ���������ֱ�Ӵ����ݼ���վ���ֶ��������ݼ������������ĸ��ļ���
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)



MNIST���ݼ����������֣�55000����������ѵ����10000���������ڲ��ԣ�5000������������֤��

��ǰ������ÿ��MNIST���ݰ���2���֣�һ����д���ֵ�ͼ���һ����Ӧ�ı�ǩ�����ǿ��Գ�ͼ��x���ͱ�ǩ��y����

ÿһ��ͼƬ��28*28���صģ����ǿ��԰����ͼƬ����Ϊһ��������飺
![title](https://leanote.com/api/file/getImage?fileId=598731b4ab6441463e00171f)

���ǿ��԰��������ƽ�̳�һ��28*28=784ά���������������ǾͿ��԰�ÿ��ͼƬ����һ��784ά�����ռ��е�һ���㡣

�����Ļ���mnist.train.images����һ��tensor����shape��[55000, 784]��
![title](https://leanote.com/api/file/getImage?fileId=598732ccab6441463e001738)

�ڱ�ʾ���У�����������ݱ�ǩ��ɡ�one-hot����������ν��one-hot��������һ��������ֻ��һ��ά����1���������0.�ڱ������У���n�����ֿ��Ա�ʾΪһ���ڵ�nά��1��һ��one-hot��������������3��Ӧ�ľ���[0,0,0,1,0,0,0,0,0,0]. ��Ӧ�ģ�mnist.train.labels����shape��[55000, 10]��һ��tensor��
![title](https://leanote.com/api/file/getImage?fileId=598733e0ab644143e40015e2)

-----------------------------
# Softmax�ع�
�������Ƕ���һ����򵥵ĵ���ȫ�������磬���㹫ʽΪ��y=Wx+b,Ȼ������softmax������Ԥ����ʣ�Ԥ��������Ķ�ӦԤ��ķ��ࡣ

![title](https://leanote.com/api/file/getImage?fileId=59873456ab6441463e00177a)

![title](https://leanote.com/api/file/getImage?fileId=59873473ab644143e40015ea)

![title](https://leanote.com/api/file/getImage?fileId=59873482ab644143e40015ec)

---------------------------
# ����ʵ��
## ����������ģ��
����import TensorFlow��

    import tensorflow as tf

����x��

    x = tf.placeholder(tf.float32, [None, 784])

��λ����W��b��

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

�������ǵ�ģ�ͣ�ֻ��Ҫһ�и㶨��

    y = tf.nn.softmax(tf.matmul(x, W) + b)

## ѵ��ģ��
���ǲ���cross-entropy��Ϊ��ʧ������
![title](https://leanote.com/api/file/getImage?fileId=598737f8ab6441463e0017d2)

���У�![title](https://leanote.com/api/file/getImage?fileId=59873e80ab6441463e00184c)������Ԥ��ķֲ���![title](https://leanote.com/api/file/getImage?fileId=59873eaaab644143e40016e1)����ʵ�ķֲ���

������ʵ���ֵ��

    y_ = tf.placeholder(tf.float32, [None, 10])

������ʧ������

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

ѡ���Ż��㷨������ѡ������ݶ��½�����

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

����Ự����������ͳ�ʼ��������

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

��ʼѵ��������

    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      
## ����ģ��

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
���׼ȷ��ԼΪ92%��

-------------------------------
# �����������£�

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

˵�����������ԣ��������ݼ����ٶȷǳ���������ֱ�Ӵ����ݼ���վ���ֶ��������ݼ������������ĸ��ļ���
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
Ȼ���������������ڵ�ͬһ��Ŀ¼�£��½��ļ���MNIST_data���������ĸ��ļ��ŵ��ļ������档

�������н����
![title](https://leanote.com/api/file/getImage?fileId=5991b7dfab644147bd002578)
�������������ģ��׼ȷ��Ϊ91.72%��

�ٷ�GitHub�����ַ��
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py

------------------------------------
# �ο�����
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/
http://yann.lecun.com/exdb/mnist/
http://www.jianshu.com/p/87581c7082ba


