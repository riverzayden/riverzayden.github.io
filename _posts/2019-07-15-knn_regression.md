---

layout: post

title:  "[ML] KNN(K-nearest neighbors) Regression "

subtitle:   "[ML] KNN(K-nearest neighbors) Regression "

categories: ml

tags: ml neighbors k-nearest regression

comments: true

img: machine_learning.png

---





#### KNN(K-nearest neighbors) Regression



* 정의 

  * 가까운 점들을 기준으로 , 점들의 평균으로 예측하는 것

* 작동방식

  * 13번 점을 예측하고자 할 경우,   k=3이다.

  

  ![knn_regression_image_1](/assets/img/machine_learning/knn_regression_image_1.PNG)

  * 선택된 점은 6, 5, 1번점이 선택되었고, 13Predict =  (77+72+60)/3 = 69.66

  ![knn_regression_image_2](/assets/img/machine_learning/knn_regression_image_2.PNG)





* 거리 계산방법

1. **Euclidean Distance:** Euclidean distance is calculated as the square root of the sum of the squared differences between a new point (x) and an existing point (y).
2. **Manhattan Distance** : This is the distance between real vectors using the sum of their absolute difference.

![knn_regression_image_3](/assets/img/machine_learning/knn_regression_image_3.PNG)

1. **Hamming Distance**: It is used for categorical variables. If the value (x) and the value (y) are same, the distance D will be equal to 0 . Otherwise D=1.

![knn_regression_image_4](/assets/img/machine_learning/knn_regression_image_4.PNG)



* Python Example

```python
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import pandas as pd


boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target ,test_size=0.2)
print(len(X_train), len(X_test))
```

```python
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
```

![knn_regression_image_5](/assets/img/machine_learning/knn_regression_image_5.PNG)



```python
#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
```

![knn_regression_image_6](/assets/img/machine_learning/knn_regression_image_6.PNG)





## 참고 자료

https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/