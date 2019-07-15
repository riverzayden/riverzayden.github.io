---

layout: post

title:  "[ML] Logistic Regression  "

subtitle:   "[ML] Logistic Regression  "

categories: ml

tags: ml Logistic Regression classification odds roc_curve

comments: true

img: machine_learning.png

---



#### Logistic Regression 



* 정의

  * 회귀분석이지면 분류성격을 갖고 있다. 
  * Regression이라해서 연속형 Y값을 예측하는 것 같지만, 범주형인 경우에 사용하는 분류 방법이다.

  ![logistic_regression_image_1](/assets/img/machine_learning/logistic_regression_image_1.PNG)

  * 왼쪽 그림의 경우 Y가 0또는 1인 경우라면 선형회귀로는 fitting하기 힘들다. 따라서 곡선으로 fitting하기 위해 사용하는 것이 로지스틱함수(로짓변환)이다.
  * odds_ratio = p/(1-p)
    * example ) 실패에 비해 생존할 확률의비 = 0.38/0.62 = 0.61  ( 백명 사망할 동한 61명 생존)
    * 이것을 로짓변환하여 사용 
  * 보통 **ROC Curve**를 그리고 **AUC (the Area Under a ROC Curve)**로 판단



* Python Example

```python
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
df = sns.load_dataset("titanic")
df.head()
```

![logistic_regression_image_2](/assets/img/machine_learning/logistic_regression_image_2.PNG)

```python
feature_names = ["pclass", "age", "sex"]
dfX = df[feature_names].copy()
dfy = df["survived"].copy()
dfX["sex"] = LabelEncoder().fit_transform(dfX["sex"])
dfX["age"].fillna(dfX["age"].mean(), inplace=True)
dfX2 = pd.DataFrame(LabelBinarizer().fit_transform(dfX["pclass"]),
                    columns=['c1', 'c2', 'c3'], index=dfX.index)
dfX = pd.concat([dfX, dfX2], axis=1)
del(dfX["pclass"])
dfX.tail()

X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfy, test_size=0.3,)
X_train.tail()
```

![logistic_regression_image_3](/assets/img/machine_learning/logistic_regression_image_3.PNG)

```python
log_clf = LogisticRegression()
log_clf.fit(X_train,Y_train)
log_clf.score(X_test, Y_test)


```

![logistic_regression_image_4](/assets/img/machine_learning/logistic_regression_image_4.PNG)

```python
feature_names = ["pclass", "age", "sex","class"]
dfX = df[feature_names].copy()
dfy = df["survived"].copy()
dfX["sex"] = LabelEncoder().fit_transform(dfX["sex"])
dfX["age"].fillna(dfX["age"].mean(), inplace=True)
dfX["class"] = dfX["class"].dropna()
dfX["class"] = LabelEncoder().fit_transform(dfX["class"])
dfX["pclass"] = LabelEncoder().fit_transform(dfX["pclass"])
X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfy, test_size=0.3)
X_train.head()
```

![logistic_regression_image_5](/assets/img/machine_learning/logistic_regression_image_5.PNG)

```python
log_clf = LogisticRegression()
log_clf.fit(X_train,Y_train)
log_clf.score(X_test, Y_test)

```

![logistic_regression_image_6](/assets/img/machine_learning/logistic_regression_image_6.PNG)

```python
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve,log_loss,auc
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
import numpy as np
import matplotlib.pyplot as plt
y_pred = log_clf.predict(X_test)
[fpr, tpr, thr] = roc_curve(Y_test, y_pred)
print('Train/Test split results:')
print(log_clf.__class__.__name__+" accuracy is %2.3f" % accuracy_score(Y_test, y_pred))
print(log_clf.__class__.__name__+" log_loss is %2.3f" % log_loss(Y_test, y_pred))
print(log_clf.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
```

![logistic_regression_image_7](/assets/img/machine_learning/logistic_regression_image_7.PNG)





## 참고 자료

https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python

[https://medium.com/qandastudy/mathpresso-%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EC%8A%A4%ED%84%B0%EB%94%94-4-%ED%9A%8C%EA%B7%80-%EB%B6%84%EC%84%9D-regression-2-4f938f1f1c2d](https://medium.com/qandastudy/mathpresso-머신-러닝-스터디-4-회귀-분석-regression-2-4f938f1f1c2d)