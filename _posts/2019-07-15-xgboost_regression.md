---

layout: post

title:  "[ML] XGBoost Regression "

subtitle:   "[ML] XGBoost Regression "

categories: ml

tags: ml xgb boosting regression

comments: true

img: machine_learning.png

---



#### XGBoost Regression



* 정의

  - 약한 분류기를 세트로 묶어서 정확도를 예측하는 기법이다.
  - 욕심쟁이(Greedy Algorithm)을 사용하여 분류기를 발견하고 분산처리를 사용하여 빠른 속도로 적합한 비중 파라미터를 찾는 알고리즘이다. 
  - 부스팅 알고리즘이 기본원리 

  

* 장점

  * 병렬 처리를 사용하기에 학습과 분류가 빠르다
  * 유연성이 좋다. 커스텀 최적화 옵션을 제공한다
  * 욕심쟁이(Greedy-algorithm)을 사용한 자동 가지치기가 가능하다. 과적합이 잘일어나지 않는다.
  * 다른 알고리즘과 연계하여 앙상블 학습이 가능하다. 

* 수식 예 

  * Y = w1 * M(x)+ w2 * G(x)+ w3 * H(x) + error   ==> 세개의 모델이 함게 적용된 것이다. 



* 파라미터
  - 일반 파라미터 - 도구의 모양을 결정
  -  부스트 파라미터 - 트리마다 가지를 칠 때 적용하는 옵션을 정의
  -  학습과정 파라미터 -  최적화 퍼모먼스를 결정
* 일반파라미터
  * booster : 어떤 부스터 구조를 쓸지 결정한다. ( gbtree, gblinear, dart)
  * nthread : 몇개의 쓰레드를 동시에 처리하도록 할지 결정한다. 디폴트는 '가능한 많이'
  * num_feature : feature차원의 숫자를 정해야하는 경우 옵션을 세팅. '디폴트는 가능한 많이'
* 부스팅파라미터
  * eta: learning rate와 같다. 트리에 가지가 많을 수록 과적합하기 쉽다. 매 부스팅 스탭마다 weight를 주어 부스팅 과정에 과적합이 일어나지 않도록 한다.
  * gamma: 정보흭득(information Gain)에서 -r로 표현한 바 있다. 이것이 커지면, 트리 깊이가 줄어들어 보수적인 모델이 된다. ( 디폴트는 0 )
  * max_depth : 한 트리의 maxium depth. 숫자가를 키울수록 보델의 복잡도가 커진다. 과적합 하기 쉽다. 디폴트는 6, 이 때 리프노트의 개수는 최대 2^6 = 64개이다.
  * lambda (L2 reg-form) : L2 Regularization Form에 달리는 weights이다. 숫자가 클수록 보수적인 모델이 된다.
  * alpha(L1 reg-form) : L1 Regularization Form에 달리는 weights이다. 숫자가 클수록 보수적인 모델이 된다.
* 학습과정 파라미터
  * objective : 목적함수이다. reg:linear(linear-regression), binary:logistic(binary-logistic-classification), count:poisson(count data poison regression) 등 다양
  * eval_metric : 모델의 평가 함수를 조정하는 함수 - rmse(root mean square error), logloss(log-likelihood),  map(mean average precision) 등 데이터의 특성에 맞게 평가 함수를 조정 
* 커멘드 라인 파라미터
  * num_rounds : 부스팅 라운드를 결정한다. 랜덤하게 생성되는 모델이니만큼 이 수 가 적당히 큰게 좋다 epochs 옵션과 동일하다 . 



* Python Example

```python
from sklearn.datasets import load_boston
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target ,test_size=0.1)
xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
print(len(X_train), len(X_test))
xgb_model.fit(X_train,y_train)
```

![xgboost_regression_image_1](/assets/img/machine_learning/xgboost_regression_image_1.PNG)

```python
xgboost.plot_importance(xgb_model)
```

![xgboost_regression_image_2](/assets/img/machine_learning/xgboost_regression_image_2.PNG)

```python
predictions = xgb_model.predict(X_test)
predictions
```

![xgboost_regression_image_3](/assets/img/machine_learning/xgboost_regression_image_3.PNG)

```python
r_sq = xgb_model.score(X_train, y_train)
print(r_sq)
print(explained_variance_score(predictions,y_test))
```

![xgboost_regression_image_4](/assets/img/machine_learning/xgboost_regression_image_4.PNG)





## 참고 자료 

https://brunch.co.kr/@snobberys/137