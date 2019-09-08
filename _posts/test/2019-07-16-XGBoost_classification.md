#### XGBoost classification



- 정의

  - 약한 분류기를 세트로 묶어서 정확도를 예측하는 기법이다.
  - 욕심쟁이(Greedy Algorithm)을 사용하여 분류기를 발견하고 분산처리를 사용하여 빠른 속도로 적합한 비중 파라미터를 찾는 알고리즘이다. 
  - 부스팅 알고리즘이 기본원리 
  - Boosting은 sequential process로 이전 tree로 부터 얻은 정보를 다음 tree를 생성하는데 활용하기 때문이다.

  

- 장점

  - 병렬 처리를 사용하기에 학습과 분류가 빠르다
  - 유연성이 좋다. 커스텀 최적화 옵션을 제공한다
  - 욕심쟁이(Greedy-algorithm)을 사용한 자동 가지치기가 가능하다. 과적합이 잘일어나지 않는다.
  - 다른 알고리즘과 연계하여 앙상블 학습이 가능하다. 



* 어떤 파라미터를 중심으로 튜닝해야 하는가?
  * General Parameter

    * booster [default : gbtree]

      부스터는 각각의 iteration에서 사용할 모델을 고르는 옵션이다. gbtree, gblinear 2가지 옵션이 있는데 각각 트리와 선형회귀 모델을 나타낸다. 사실 gbtree 모델이 거의 항상 gblinear보다 좋은 성능을 나타내기 때문에 수정할 필요가 없다고 한다.

    * silent [default = 0]

      이 옵션을 True로 설정하는 경우에는 running message가 출력되지 않는데, GBM은 튜닝에 시간이 많이 걸리기도 하고 모델이 적합되는 과정을 이해하기 위해서는 False로 설정해 놓는 것이 좋다.
  
  * Booster Parameter (For Tree Model)
  
    Booster Parameter는 12가지가 존재하는데, 위의 사이트에서는 반드시 튜닝해야 할 파라미터를 min_child_weight와 max_depth, gamma 정도로 이야기 하고 있다.
  
    * min_child_weight [default = 1]
  
      overfitting을 컨트롤하는 파라미터로, 값이 높아지면 under-fitting 되는 경우가 있기 때문에 CV를 통해 튜닝되어야 한다고 한다.
  
    * max_depth [default = 6]
  
      트리의 최대 깊이를 정의하는 파라미터로 Typical Value는 3-10 정도라고 하므로, 마찬가지로 CV를 통해 튜닝되어야 한다.
  
    * gamma [default = 0]
  
      노드가 split 되기 위한 loss function의 값이 감소하는 최소값을 정의한다. gamma 값이 높아질 수록 알고리즘은 보수적으로 변하고 loss function의 정의에 따라 적정값이 달라지기 때문에 반드시 튜닝되어야 한다.
  
    * objective [default = reg:linear]
  
      옵션은 ‘reg:linear’, ‘binary:logistic’, ‘multi:softmax’, ‘multi:softprob’ 4가지가 존재한다. 회귀의 경우 reg, binary 분류의 경우 binary, 다중분류의 경우 multi 옵션을 사용하면 되는데, multi:softmax의 경우는 분류된 class를 return하고, multi:softprob의 경우는 각 class에 속할 확률을 return한다.
  
    * ```python
      XGBClassifier.fit(X, y, sample_weight=None, eval_set=None, eval_metric=None,
                        early_stopping_rounds=None, verbose=True, xgb_model=None)
      ```
  
    * eval_metric
  
      validation set에 적용되는 모델 선택 기준이 된다. 회귀의 경우는 ‘rmse’가 기본 옵션이고, 분류 모델의 경우는 ‘error’ (오분류율)이 기본 옵션이다. 또한, 사용가능한 옵션은 ‘mae’, ‘logloss’, ‘merror’, ‘mlogloss’, ‘auc’가 있는데 이 이외의 옵션을 사용하기 위해서는 다음과 같이 함수를 정의하여야 한다. 분류 모델 경우의 F1-Score를 eval metric으로 사용하는 옵션의 경우 함수를 다음과 같이 구현할 수 있다.
  
      ```python
      from sklearn.metrics import f1_score
      
      def xgb_f1(y,t):
          t = t.get_label()
          threshold_value=0.5
          y_bin = [1 if y_cont > threshold_value else 0 for y_cont in y] 
          return 'f1',f1_score(t,y_bin)
      ```
  
      
  
      우선 함수는 2개의 인자를 받아야 한다. 첫 번째 인자는 모델이 예측한 값을 포함하고 있고, 두 번째 인자는 validation set의 해답 정보를 포함하고 있다. 첫 번째 인자 안의 y_cont 변수는 모델이 예측한 각 데이터가 참일 확률이다. 기존 모델은 참일 확률이 0.5 이상인 경우만 참으로 분류하지만, 위와 같이 변수를 추가하여 threshold 값도 직접 튜닝해줄 수 있다.
  
      ```python
      clf.fit(X_train, Y_train,
              eval_set=[(X_train, Y_train),(X_valid, Y_valid)], #모델에서 자체적으로 평가에 사용할 데이터
              eval_metric=xgb_inv_f1, #모델의 목적함수 지정(최소화할 목적함수 1-f1_score)
              early_stopping_rounds=100, #1o0 Interations 동안 최대화 되지 않으면 stop
              verbose=10) #Iteration 과정을 10 단위로 보여준다.
      ```
  
      
  
      아쉽지만 기존 모델에는 Learning Task Parameter에 maximize 옵션이 존재하는데, Scikit-Learn으로 wrapped된 XGBoostClassifier와 XGBoostRegressor의 경우에는 maximize 옵션이 존재하지 않는다. F1-score은 최대화 될 수록 좋은 것인데 XGBoostClassifier와 XGBoostRegressor의 경우에는 둘 다 eval_metric을 최소화하려고 하기 때문에, 함수를 1-F1_Score 형태로 정의해줘야 하는 번거로움이 발생한다.
  
      eval_set에는 Train 데이터와 Valid 데이터를 튜플 형태로 넣어주면 된다.

* Python Example

```python
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# list for column headers
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# open file with pd.read_csv
dataset = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", names=names)
print(dataset.shape)
# print head of data set
dataset.head()
```

![xgboost_classification_image_1](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\xgboost_classification_image_1.PNG)

```python

# split data into X and y
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]
# split data into train and test sets
seed = 1
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

```

![xgboost_classification_image_2](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\xgboost_classification_image_2.PNG)

## 참고 자료 

https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

https://jungsooyun.github.io/misc/2018/02/19/XGBoost.html

https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/