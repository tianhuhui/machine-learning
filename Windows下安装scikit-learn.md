## scikit-learn����
scikit-learn��Python��һ����Դ����ѧϰģ�飬��������NumPy��SciPy��matplotlibģ��֮�ϣ���ϸ���ܿ��Բ���scikit-learn������http://scikit-learn.org/stable/

-------------------------------------
## ��װ׼������
- Python
- Numpy

NumPy��Python��һ�ֿ�Դ����ֵ������չ���ߣ��������洢�ʹ�����;��󣬱�Python�����Ƕ���б�nested list structure���ṹҪ��Ч�Ķࣨ�ýṹҲ����������ʾ���󣩡�

- Scipy

SciPy��һ���ġ�����ʹ�õġ�ר��Ϊ��ѧ�͹�����Ƶ�Python���߰���������ͳ�ơ��Ż������ϡ����Դ���ģ�顢����Ҷ�任���źź�ͼ������΢�ַ���������ȡ�

- matplotlib

matplotlib��python�������Ļ�ͼ�⣬���ṩ��һ���׺�MATLAB��ϸ������API��ʮ���ʺϽ���ʽ��ͼ�����ҿ��Է���ؽ�����Ϊ��ͼ�ؼ���Ƕ�뵽GUIӦ�ó����С�

### ���ص�ַ
Python��https://www.python.org/downloads/
NumPy��https://pypi.python.org/pypi/numpy
��or http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy�������� mkl ��δ���� mkl ��numpy�������ϻ����������죬�����ǰ��������ַ��װ������ mkl ��� numpy��
SciPy��http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
matplotlib��http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib
sciki-learn��http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn

���Լ���װ�İ汾���£�
- Python��3.6.2��64bit��
- NumPy��numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl
- SciPy��scipy-0.19.1-cp36-cp36m-win_amd64.whl
- matplotlib��matplotlib-2.0.2-cp36-cp36m-win_amd64.whl
- scikit-learn��scikit_learn-0.18.2-cp36-cp36m-win_amd64.whl

�ļ�����ɣ�
**����-��汾��-python�汾-ƽ̨**

���磺
scikit_learn-0.18.2-cp36-cp36m-win_amd64.whl
������ǣ�
scikit-learn�汾��0.18.2
cp36����python3.6
win_amd64����Windows 64λϵͳ

--------------
## ��װ����
���Ȱ�װPython���ǵù�ѡ���뵽����������
Ȼ�����ΰ�װNumPy��SciPy��matplotlib��

- ��װNumPy��
pip install ".\numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl"
![title](https://leanote.com/api/file/getImage?fileId=597345b9ab644127c700170c)

��װ�ɹ���
���������XXX is not a supported wheel on this platform������ʾ������ϸ�����python�汾��whl�汾�Ƿ�һ�£���

- ��װSciPy��
pip install .\scipy-0.19.1-cp36-cp36m-win_amd64.whl
![title](https://leanote.com/api/file/getImage?fileId=597346e8ab644127c7001726)

- ��װmatplotlib
pip install .\matplotlib-2.0.2-cp36-cp36m-win_amd64.whl
![title](https://leanote.com/api/file/getImage?fileId=59734771ab644127c7001732)

- ��װscikit-learn
pip install .\scikit_learn-0.18.2-cp36-cp36m-win_amd64.whl
![title](https://leanote.com/api/file/getImage?fileId=597347c9ab644125a100168b)

-----------------------
## ����scikit-learn
���Ǹ���scikit-learn������ָ����http://scikit-learn.org/stable/tutorial/basic/tutorial.html�����������Ƿ�װ�ɹ��������Ƿ���á�

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