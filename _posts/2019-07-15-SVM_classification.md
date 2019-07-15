---

layout: post

title:  "[ML] SVM ( Support Vector Machine)  "

subtitle:   "[ML] SVM ( Support Vector Machine)  "

categories: ml

tags: ml svm support vector classification

comments: true

img: machine_learning.png

---



#### SVM ( Support Vector Machine) 



* 정의

  * 분리된 초평면에 의해 정의된 분류 모델이다.
  * 최적의 초평면을 찾는 것 
  * 가장 최적의 의사 결정 경계는 모든 클래스의 가장 가까운 점으로부터 최대 마진을 갖는 결정 경계입니다. 
  * 결정 경계와 점 사이의 거리를 최대화하는 결정 경계로부터의 가장 가까운 점을 그림 2에서 보듯이 Support Vector 라고 부른다. Support Vector 의 결정 경계는 최대 마진 분류기 또는 최대 마진 하이퍼 평면이라고 불린다. 

  ![svm_image_1](/assets/img/machine_learning/svm_image_1.PNG)

  * python에서 svm kernel의 종류 ==> linear, poly(다항), rbf(가우시안), sigmoid(시그모이드)



* Python Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  
import pandas as pd
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame({'class':iris.target})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
```



```python

svclassifier = SVC(kernel='linear', degree=8)  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  
print('score : ',svclassifier.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
```

![svm_image_2](/assets/img/machine_learning/svm_image_2.PNG)

```python

svclassifier = SVC(kernel='poly', degree=8)  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  
print('score : ',svclassifier.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

```

![svm_image_3](/assets/img/machine_learning/svm_image_3.PNG)

```python

svclassifier = SVC(kernel='rbf', degree=8)  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  
print('score : ',svclassifier.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
```

![svm_image_4](/assets/img/machine_learning/svm_image_4.PNG)



```python

svclassifier = SVC(kernel='sigmoid', degree=8)  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  
print('score : ',svclassifier.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
```

![svm_image_5](/assets/img/machine_learning/svm_image_5.PNG)





## 참고 자료

https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/