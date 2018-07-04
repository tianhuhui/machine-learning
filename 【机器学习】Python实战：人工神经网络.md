## MLPClassifier
Class MLPClassifier implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
��http://scikit-learn.org/stable/modules/neural_networks_supervised.html��

���ڶ�������磬scikit-learn�ṩ��MLPClassifier�࣬���ʼ������Ϊ��
class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

## ����������Ӧ��ʵ��
### ��������
���β�����з��ࡣ�β�����ݼ�һ����150�����ݣ���Щ���ݷ�Ϊ3�ࣨ�ֱ�Ϊsetosa��versicolor��virginica����ÿ��50�����ݡ�ÿ�����ݰ���4�����ԣ���Ƭ��sepal�����ȡ���Ƭ��ȡ����꣨petal�����ȡ������ȡ�

### ����

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors, datasets
    from sklearn.neural_network import MLPClassifier
    
    #�������ݼ�
    np.random.seed(0)
    iris = datasets.load_iris()
    X=iris.data[:, 0:2] #����ʹ�õ���sepal length��sepal width��������������Ϊ���Է�����ڶ�άͼ���ϱ��ֳ���
    Y=iris.target #���ֵ
    data = np.hstack((X, Y.reshape(Y.size, 1)))
    np.random.shuffle(data) # ��ϴ���ݡ���ΪĬ�ϵ�iris ���ݼ���ǰ50�����������0���м�50�����������1��ĩβ50�����������2.��ϴ���������˳��
    X=data[:, :-1]
    Y=data[:, -1]
    train_x =X[:-30]
    test_x = X[-30:]
    train_y = Y[:-30]
    test_y = Y[-30:]
    
    def plot_classifier_predict_meshgrid(ax, clf, x_min, x_max, y_min, y_max):
        '''
        ����MLPClassifier�ķ�����
         
        :param ax: Axesʵ�������ڻ�ͼ 
        :param clf: MLPClassifierʵ��
        :param x_min: ��һά��������Сֵ
        :param x_max: ��һά���������ֵ
        :param y_min: �ڶ�ά��������Сֵ
        :param y_max: �ڶ�ά���������ֵ
        :return: None
        '''
        plot_step = 0.02 #����
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)
    
    def plot_samples(ax, x, y):
        '''
        ���ƶ�ά���ݼ�
        
        :param ax: Axesʵ�������ڻ�ͼ 
        :param x: ��һά����
        :param y: �ڶ�ά����
        :return: None
        '''
        n_classes = 3
        plot_colors = "bry"
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            ax.scatter(x[idx, 0], x[idx, 1], c = color, label = iris.target_names[i], cmap=plt.cm.Paired)
    
    def mlpclassifier_iris():
        '''
        ʹ��MLPClassifierԤ��������iris���ݼ�
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
        
    
### ���н��
![mlpclassifier_iris���н��](https://leanote.com/api/file/getImage?fileId=5991bc3eab644147bd0025f8)

����ͼ��ʾ����������ѵ�������ϵ�Ԥ�⾫��Ϊ80.8333%���ڲ��Լ��ϵ�Ԥ�⾫��Ϊ80.0%��
    