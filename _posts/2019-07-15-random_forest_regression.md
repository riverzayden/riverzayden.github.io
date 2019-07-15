#### Random Forest Regression

* 정의

  랜덤 포래스트는 앙상블 기법 중 하나이다. 

![random_forest_regression_image_1](D:\HBEE회사\python자료\정리본\md_image\random_forest_regression_image_1.PNG)



* Python Example

``` python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plot
data = load_wine()   # data load 
data.target[[10, 80, 140]]
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.columns)
df.head()
```

![random_forest_regression_image_2](D:\HBEE회사\python자료\정리본\md_image\random_forest_regression_image_2.PNG)

```python
X = df.iloc[:,1:].to_numpy()
Y = df.iloc[:,0].to_numpy()
wineNames = np.array(data.feature_names[1:])
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3, random_state = 531)
print(len(xTrain), len(xTest))
```



```python
mseOos = []
nTreeList = range(50, 500, 10)
for iTrees in nTreeList:
    depth = None
    maxFeat = 4 #조정해볼 것
    wineRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees,
                    max_depth=depth, max_features=maxFeat,
                    oob_score=False, random_state=531)
    wineRFModel.fit(xTrain, yTrain)
    #데이터 세트에 대한 MSE 누적
    prediction = wineRFModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))
print("MSE")
print(mseOos)
```

![random_forest_regression_image_3](D:\HBEE회사\python자료\정리본\md_image\random_forest_regression_image_3.PNG)

```python

plot.plot(nTreeList, mseOos)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
#plot.ylim([0.0, 1.1*max(mseOob)])
plot.show()
 
#피처 중요도 도표 그리기
featureImportance = wineRFModel.feature_importances_
 
#가장 높은 중요도 기준으로 스케일링
featureImportance = featureImportance/featureImportance.max()
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0])+.5
plot.barh(barPos, featureImportance[sorted_idx], align='center')
plot.yticks(barPos, wineNames[sorted_idx])
plot.xlabel('Variable Importance')
plot.show()

```

![random_forest_regression_image_4](D:\HBEE회사\python자료\정리본\md_image\random_forest_regression_image_4.PNG)



```python

regr = RandomForestRegressor(max_depth=4, random_state=531,
                          n_estimators=150)
regr.fit(xTrain, yTrain)
prediction = regr.predict(xTest)
print(mean_squared_error(yTest, prediction))

featureImportance = regr.feature_importances_
 
#가장 높은 중요도 기준으로 스케일링
featureImportance = featureImportance/featureImportance.max()
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0])+.5
plot.barh(barPos, featureImportance[sorted_idx], align='center')
plot.yticks(barPos, wineNames[sorted_idx])
plot.xlabel('Variable Importance')
plot.show()

```

![random_forest_regression_image_5](D:\HBEE회사\python자료\정리본\md_image\random_forest_regression_image_5.PNG)





## 참고 자료

http://blog.naver.com/PostView.nhn?blogId=navdps&logNo=220621098783

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html