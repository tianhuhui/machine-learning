## 线性回归模型
### 问题描述
在线性回归问题中，使用的数据集是scikit-learn自带的一个糖尿病病人的数据集。该数据集从糖尿病病人采样并整理后，特点如下：
- 数据集有442个样本
- 每个样本有10个特征
- 每个特征都是浮点数，数据都在-0.2~0.2之间
- 样本的目标在整数25~346之间

### 代码

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model,cross_validation
    
    def load_data():
        '''
        加载用于回归问题的数据集
    
        :return: 一个元组，用于回归问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
        '''
        diabetes = datasets.load_diabetes()#使用 scikit-learn 自带的一个糖尿病病人的数据集
        return cross_validation.train_test_split(diabetes.data,diabetes.target,
    		test_size=0.25,random_state=0) # 拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
    def test_LinearRegression(*data):
        '''
        测试 LinearRegression 的用法
    
        :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
        :return: None
        '''
        X_train,X_test,y_train,y_test=data
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
        print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test) - y_test) ** 2))
        print('Score: %.2f' % regr.score(X_test, y_test))
    if __name__=='__main__':
        X_train,X_test,y_train,y_test=load_data() # 产生用于回归问题的数据集
        test_LinearRegression(X_train,X_test,y_train,y_test) # 调用 test_LinearRegression
        
输出结果如下：
Coefficients:[ -43.26774487 -208.67053951  593.39797213  302.89814903 -560.27689824
  261.47657106   -8.83343952  135.93715156  703.22658427   28.34844354], intercept 153.07
Residual sum of squares: 3180.20
Score: 0.36

-----------------------------
## 线性回归模型的正则化
- Ridge Regression
- Lasso Regression
- Elastic Net

--------------------------------
## 逻辑回归模型
scikit-learn中的LogisticRegression实现了逻辑回归模型，原型如下：
class sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)[source]

### 问题描述
使用逻辑回归模型，对鸢尾花进行分类。鸢尾花数据集一共有150个数据，这些数据分为3类（分别为setosa、versicolor、virginica），每类50个数据。每个数据包含4个属性：萼片（sepal）长度、萼片宽度、花瓣（petal）长度、花瓣宽度。

### 代码

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model,cross_validation
    
    def load_data():
        '''
        加载用于分类问题的数据集
    
        :return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
        '''
        iris=datasets.load_iris() # 使用 scikit-learn 自带的 iris 数据集
        X_train=iris.data
        y_train=iris.target
        return cross_validation.train_test_split(X_train, y_train,test_size=0.25,
    		random_state=0,stratify=y_train)# 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
    def test_LogisticRegression(*data):
        '''
        测试 LogisticRegression 的用法
    
        :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
        :return: None
        '''
        X_train,X_test,y_train,y_test=data
        regr = linear_model.LogisticRegression()
        regr.fit(X_train, y_train)
        print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_))
        print('Score: %.2f' % regr.score(X_test, y_test))
    def test_LogisticRegression_multinomial(*data):
        '''
        测试 LogisticRegression 的预测性能随 multi_class 参数的影响
    
        :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
        :return: None
        '''
        X_train,X_test,y_train,y_test=data
        regr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
        regr.fit(X_train, y_train)
        print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_))
        print('Score: %.2f' % regr.score(X_test, y_test))
    def test_LogisticRegression_C(*data):
        '''
        测试 LogisticRegression 的预测性能随  C  参数的影响
    
        :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
        :return: None
        '''
        X_train,X_test,y_train,y_test=data
        Cs=np.logspace(-2,4,num=100)
        scores=[]
        for C in Cs:
            regr = linear_model.LogisticRegression(C=C)
            regr.fit(X_train, y_train)
            scores.append(regr.score(X_test, y_test))
        ## 绘图
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(Cs,scores)
        ax.set_xlabel(r"C")
        ax.set_ylabel(r"score")
        ax.set_xscale('log')
        ax.set_title("LogisticRegression")
        plt.show()
    
    if __name__=='__main__':
        X_train,X_test,y_train,y_test=load_data() # 加载用于分类的数据集
        test_LogisticRegression(X_train,X_test,y_train,y_test) # 调用  test_LogisticRegression
        test_LogisticRegression_multinomial(X_train,X_test,y_train,y_test) # 调用  test_LogisticRegression_multinomial
        test_LogisticRegression_C(X_train,X_test,y_train,y_test) # 调用  test_LogisticRegression_C

----------------------        
### 参考资料
华校专：《Python大战机器学习》
周志华：《机器学习》