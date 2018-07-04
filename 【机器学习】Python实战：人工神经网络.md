## MLPClassifier
Class MLPClassifier implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
（http://scikit-learn.org/stable/modules/neural_networks_supervised.html）

对于多层神经网络，scikit-learn提供了MLPClassifier类，其初始化函数为：
class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

## 多层神经网络的应用实例
### 问题描述
对鸢尾花进行分类。鸢尾花数据集一共有150个数据，这些数据分为3类（分别为setosa、versicolor、virginica），每类50个数据。每个数据包含4个属性：萼片（sepal）长度、萼片宽度、花瓣（petal）长度、花瓣宽度。

### 代码

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors, datasets
    from sklearn.neural_network import MLPClassifier
    
    #加载数据集
    np.random.seed(0)
    iris = datasets.load_iris()
    X=iris.data[:, 0:2] #这里使用的是sepal length和sepal width这两个特征。因为可以方便地在二维图像上表现出来
    Y=iris.target #标记值
    data = np.hstack((X, Y.reshape(Y.size, 1)))
    np.random.shuffle(data) # 混洗数据。因为默认的iris 数据集：前50个数据是类别0，中间50个数据是类别1，末尾50个数据是类别2.混洗将打乱这个顺序
    X=data[:, :-1]
    Y=data[:, -1]
    train_x =X[:-30]
    test_x = X[-30:]
    train_y = Y[:-30]
    test_y = Y[-30:]
    
    def plot_classifier_predict_meshgrid(ax, clf, x_min, x_max, y_min, y_max):
        '''
        绘制MLPClassifier的分类结果
         
        :param ax: Axes实例，用于绘图 
        :param clf: MLPClassifier实例
        :param x_min: 第一维特征的最小值
        :param x_max: 第一维特征的最大值
        :param y_min: 第二维特征的最小值
        :param y_max: 第二维特征的最大值
        :return: None
        '''
        plot_step = 0.02 #步长
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)
    
    def plot_samples(ax, x, y):
        '''
        绘制二维数据集
        
        :param ax: Axes实例，用于绘图 
        :param x: 第一维特征
        :param y: 第二维特征
        :return: None
        '''
        n_classes = 3
        plot_colors = "bry"
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            ax.scatter(x[idx, 0], x[idx, 1], c = color, label = iris.target_names[i], cmap=plt.cm.Paired)
    
    def mlpclassifier_iris():
        '''
        使用MLPClassifier预测调整后的iris数据集
        :return: None 
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
        classifier = MLPClassifier(activation='logistic', max_iter=10000, hidden_layer_sizes=(30,))
        classifier.fit(train_x, train_y)
        train_score = classifier.score(train_x, train_y)
        test_score = classifier.score(test_x, test_y)
        x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
        y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
    
        plot_classifier_predict_meshgrid(ax, classifier, x_min, x_max, y_min, y_max)
        plot_samples(ax, train_x, train_y)
        ax.legend(loc='best')
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title("train score:%f;test score:%f"%(train_score, test_score))
        plt.show()
    
    if __name__ == '__main__':
        mlpclassifier_iris()
        
    
### 运行结果
![mlpclassifier_iris运行结果](https://leanote.com/api/file/getImage?fileId=5991bc3eab644147bd0025f8)

如上图所示，分类器在训练数据上的预测精度为80.8333%，在测试集上的预测精度为80.0%。
    