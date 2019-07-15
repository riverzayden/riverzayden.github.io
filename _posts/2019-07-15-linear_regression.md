---

layout: post

title:  "[ML] Linear Regression  "

subtitle:   "[ML] Linear Regression  "

categories: ml

tags: ml linear regression

comments: true

img: 

---



#### Linear Regression 

* 정의

  * 독립변수 x와 종속변수 y간의 관계를 정량적으로 찾아내는 작업 

  * 만약 독립 변수 x와 이에 대응하는 종속 변수 y간의 관계가 다음과 같은 선형 함수 f(x)이면 **선형 회귀분석(linear regression analysis)**이라고 한다.

    ![linear_regression_image_1](/assets/img/machine_learning/linear_regression_image_1.PNG)

* Python Example 

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
boston = load_boston()
model_boston = LinearRegression(fit_intercept=True).fit(boston.data, boston.target)
model_boston
```

![linear_regression_image_2](/assets/img/machine_learning/linear_regression_image_2.PNG)

```python
print('slope:', model_boston.coef_)
print('----')
print('intercept', model_boston.intercept_)
r_sq = model_boston.score(boston.data, boston.target)
print('coefficient of determination:', r_sq)
```

![linear_regression_image_3](/assets/img/machine_learning/linear_regression_image_3.PNG)

```python
import matplotlib.pyplot as plt
predictions = model_boston.predict(boston.data)

plt.scatter(boston.target, predictions)
plt.xlabel(u"Real Housing Price")
plt.ylabel(u"Predict Price")
plt.title("")
plt.show()
```

![linear_regression_image_4](/assets/img/machine_learning/linear_regression_image_4.PNG)

```python
import numpy as np
import statsmodels.api as sm
x = sm.add_constant(boston.data)
stats_model = sm.OLS(boston.target, x)
stats_model = stats_model.fit()
print(stats_model.summary())
```

![linear_regression_image_5](/assets/img/machine_learning/linear_regression_image_5.PNG)

```python
print('coefficient of determination:', stats_model.rsquared)
print('adjusted coefficient of determination:', stats_model.rsquared_adj)
print('regression coefficients:', stats_model.params)

```

![linear_regression_image_6](/assets/img/machine_learning/linear_regression_image_6.PNG)

## 참고 자료 

https://realpython.com/linear-regression-in-python/