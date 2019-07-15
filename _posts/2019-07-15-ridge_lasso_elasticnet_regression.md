---

layout: post

title:  "[ML] Regularization method - Regression  "

subtitle:   "[ML] Regularization method - Regression  "

categories: ml

tags: ml regularization regression ridge lasso elastic_net

comments: true

img: machine_learning.png

---





### Regularization method - Regression 

#### Ridge Regression



* 정의 

  * Ridge 회귀모형에서는 가중치들의 제곱합(squared sum of weights)을 최소화하는 것을 추가적인 제약 조건으로 한다.

    ![ridge_lasso_elasticnet_image_1](/assets/img/machine_learning/ridge_lasso_elasticnet_image_1.PNG)

  * λ는 기존의 잔차 제곱합과 추가적 제약 조건의 비중을 조절하기 위한 하이퍼 모수(hyper parameter)이다. λ가 크면 정규화 정도가 커지고 가중치의 값들이 작아진다. λ가 작아지면 정규화 정도가 작아지며 λ 가 0이 되면 일반적인 선형 회귀모형이 된다.

* python Example

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
boston = load_boston()
X = boston.data
y = boston.target

```



```python
ridge = Ridge()
alphas = np.logspace(-4, 0, 200)
parameters = {'alpha': alphas }
ridge_reg = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error',cv=5)
ridge_reg.fit(X,y)
print(ridge_reg.best_params_)
print(ridge_reg.best_score_)
```

![ridge_lasso_elasticnet_image_4](/assets/img/machine_learning/ridge_lasso_elasticnet_image_4.PNG)





#### Lasso Regression

* 정의

  * Lasso(Least Absolute Shrinkage and Selection Operator) 회귀모형은 가중치의 절대값의 합을 최소화하는 것을 추가적인 제약 조건으로 한다.

  ![ridge_lasso_elasticnet_image_2](/assets/img/machine_learning/ridge_lasso_elasticnet_image_2.PNG)

* python Example

```python
## lasso

alphas = np.logspace(-4, 0, 200)

train_scores = []
test_scores = []
for alpha in alphas:
    model = Lasso(alpha=alpha)
    train_score = -mean_squared_error(y, model.fit(X, y).predict(X))
    test_score = np.mean(cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    train_scores.append(train_score)
    test_scores.append(test_score)

optimal_alpha = alphas[np.argmax(test_scores)]
optimal_score = np.max(test_scores)

plt.plot(alphas, test_scores, "-", label="test ")
plt.plot(alphas, train_scores, "--", label="train")
plt.axhline(optimal_score, linestyle=':')
plt.axvline(optimal_alpha, linestyle=':')
plt.scatter(optimal_alpha, optimal_score)
plt.title("Best Regularization")
plt.ylabel('score')
plt.xlabel('Regularization')
plt.legend()
plt.show()
```

![ridge_lasso_elasticnet_image_5](/assets/img/machine_learning/ridge_lasso_elasticnet_image_5.PNG)

```python

lasso = Lasso()

alphas = np.logspace(-4, 0, 200)
parameters = {'alpha': alphas }
lasso_reg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error',cv=5)
lasso_reg.fit(X,y)
print(lasso_reg.best_params_)
print(lasso_reg.best_score_)
```

![ridge_lasso_elasticnet_image_6](/assets/img/machine_learning/ridge_lasso_elasticnet_image_6.PNG)



#### Elastic Net Regression

* 정의
  * Elastic Net 회귀모형은 가중치의 절대값의 합과 제곱합을 동시에 제약 조건으로 가지는 모형이다.
* Python Example

```python

elasticnet = ElasticNet()

alphas = np.logspace(-4, 0, 200)
parameters = {'alpha': alphas }
elasticnet_reg = GridSearchCV(elasticnet, parameters, scoring='neg_mean_squared_error',cv=5)
elasticnet_reg.fit(X,y)
print(elasticnet_reg.best_params_)
print(elasticnet_reg.best_score_)
```

![ridge_lasso_elasticnet_image_7](/assets/img/machine_learning/ridge_lasso_elasticnet_image_7.PNG)







## 참고 자료

https://datascienceschool.net/view-notebook/83d5e4fff7d64cb2aecfd7e42e1ece5e/

https://towardsdatascience.com/how-to-perform-lasso-and-ridge-regression-in-python-3b3b75541ad8