#### Naive Bayes Classification

* 정의
  * Bayes 법칙에 기반한 분류기
  * Naive라는 형용사가 붙은 이유는 분류를 쉽고 빠르게 하기 위해 분류기에 사용하는 특징들이 서로 확률적으로 독립이라는 다소 순진하고 억지스러운 성격이 들어 갔기 때문
  * 특징들이 서로 독립이라는 가정에 위반하면 에러가 발생할 수 있음
  * 특징들이 너무 많은 경우에 연관관계를 모두 고려하게 되면 너무 복잡해지는 경향이 있어서 단순화 시켜 쉽고 빠르게 **판단**을 내릴때 주로 사용 
  * 모든 요소에 대한 확률을 계산한다.
  * 피처가 독립적이라고 가정
* 주로 사용 목적
  * 실시간 예측
  * 텍스트 분류 / 스팸 필터링
  * 추천 시스템
* 장점
  * 그룹이 여러개 있는 multi-class 분류에서 쉽고 빠르게 예측이 가능
  * 독립이라는 가정이 유효하다면 logistic regression과 같은 다른 방식에 비해 훨씬 결과 좋다.
  * 데이터도 적게 필요
* 단점
  * 학습데이터에는 없고, 테스트데이터에는 있는 범주에서는 확률이 0이 되어 예측 불가능 (Zero Frequency)
  * 실제 적용에서는 완전하게 독립적인 상황이 많지 않아, 사용에 어려움 

* 공식
  $$
  P(Cj|d) = P(d|Cj) * P(Cj) / P(d)
  $$
  

  * $$
    P(Cj|d) = 특정 개체 d가 특정 그룹c에 속할 사후 확률 ( Posterior Probability )
    $$

    

  * $$
    P(d|Cj) = 특정 그룹 c인 경우 d가 그룹에 속할 조건부 확률 (Likelihood)
    $$

    

  * $$
    P(Cj) = 특정 그룹c가 발생할 빈도, 즉 클래스 사전 고유 확률 (Class Prior Probability)
    $$

    

  * $$
    P(d) = 특정 개체가 발생할 확률 ( Predictor Prior Probability)
    $$



* 계산방법 ( 예제 )

![naive_bayes_classification_image_1](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_1.PNG)

- - Drew라는 이름을 갖는 사람이 있다. Drew라는 이름을 사용하는 남자의 그룹을 c1, 여자그룹을 c2라고 할 경우에, Drew라는 사람이 c1,c2에 속할 확률은 어떻게 될것인가?

  ![naive_bayes_classification_image_2](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_2.PNG)

  - ![naive_bayes_classification_image_3](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_3.PNG)

  - 위 경찰관은 여자일까? 남자일까?

    P(male | drew ) = 1/3 * 3/8 = 0.125

    P( female | drew ) = 2/5 * 5/8 = 0.25

    결과적으로 여성쪽이 높은 값이 나와서 여성일 경우가 많다.



* 분포 

  * Gaussian Naive Bayes  => 가장 기본적으로 사용하는 Naive Bayes

  ![naive_bayes_classification_image_4](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_4.PNG)

  * Bernoulli Naive Bayes ==> 모든 변수가 0 또는 1을 가져야 한다.  ( 라플라스 스무딩을 이용하여 조정할 수도 있다. )

  ![naive_bayes_classification_image_5](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_5.PNG)

  * Multinomial Naive Bayes

  ![naive_bayes_classification_image_6](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_6.PNG)

  * Complement Naive Bayes



* Python Example

```python
from sklearn import datasets
import pandas as pd
#Import Gaussian Naive Bayes model
from sklearn import naive_bayes 
from sklearn import metrics
from sklearn.model_selection import train_test_split  
#Load dataset
wine = datasets.load_wine()
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test

data = pd.DataFrame(wine.data, columns = wine.feature_names)
data.head()
```

![naive_bayes_classification_image_7](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_7.PNG)

```python
# print the names of the 13 features
print ("Features: ", wine.feature_names)

print('')
# print the label type of wine(class_0, class_1, class_2)
print ("Labels: ", wine.target_names)
print('')
print( "Data shpe : ", wine.data.shape)
```

![naive_bayes_classification_image_8](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_8.PNG)

```python

#Create a Gaussian Classifier
gnb = naive_bayes.GaussianNB()
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Gaussian Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Create a BernoulliNB Classifier
gnb = naive_bayes.BernoulliNB()
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("BernoulliNB Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Create a MultinomialNB Classifier
gnb = naive_bayes.MultinomialNB()
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Create a ComplementNB Classifier
gnb = naive_bayes.ComplementNB()
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("ComplementNB Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

![naive_bayes_classification_image_9](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_9.PNG)



```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

news = fetch_20newsgroups(subset="all")
X = news.data
y = news.target


model1 = Pipeline([
    ('vect', CountVectorizer()),
    ('model', MultinomialNB()),
])
model2 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('model', MultinomialNB()),
])
model3 = Pipeline([
    ('vect', TfidfVectorizer(stop_words="english")),
    ('model', MultinomialNB()),
])
model4 = Pipeline([
    ('vect', TfidfVectorizer(stop_words="english",
                             token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b")),
    ('model', MultinomialNB()),
])
X[0]
```

![naive_bayes_classification_image_10](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_10.PNG)

```python
%%time
for i, model in enumerate([model1, model2, model3, model4]):
    scores = cross_val_score(model, X, y, cv=5)
    print(("Model{0:d}: Mean score: {1:.3f}").format(i + 1, np.mean(scores)))
```

![naive_bayes_classification_image_11](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\naive_bayes_classification_image_11.PNG)







## 참고 자료 

https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-1-theory-8b9e361897d5

https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475

https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220867768192&proxyReferer=https%3A%2F%2Fwww.google.com%2F

http://www.cs.ucr.edu/~eamonn/CE/Bayesian%20Classification%20withInsect_examples.pdf

https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

https://datascienceschool.net/view-notebook/c19b48e3c7b048668f2bb0a113bd25f7/

https://scikit-learn.org/stable/modules/naive_bayes.html

https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html