---

layout: post

title:  "[ML] Decision Tree Regression"

subtitle:   "[ML] Decision Tree Regression"

categories: ml

tags: ml regression decision tree

comments: true

img: 

---

#### Decision Tree Regression 

![1563155317500](/assets/img/machine_learning/Decision_Tree_Regression_image_1.PNG)

* 정의 
  * **의사 결정 나무(decision tree)**는 여러 가지 규칙을 순차적으로 적용하면서 독립 변수 공간을 분할하는 분류 모형이다. 분류(classification)와 회귀 분석(regression)에 모두 사용될 수 있기 때문에 **CART(Classification And Regression Tree)**라고도 한다.
  * 전체 학습 데이터 집합(부모 노드)을 해당 독립 변수의 값이 기준값보다 작은 데이터 그룹(자식 노드 1)과 해당 독립 변수의 값이 기준값보다 큰 데이터 그룹(자식 노드 2)으로 나눈다.
  * 각각의 자식 노드에 대해 1~2의 단계를 반복하여 하위의 자식 노드를 만든다. 단, 자식 노드에 한가지 클래스의 데이터만 존재한다면 더 이상 자식 노드를 나누지 않고 중지한다.
  * 자식 노드 나누기를 연속적으로 적용하면 노드가 계속 증가하는 나무(tree)와 같은 형태
  
* example
  * ![1563155317500](/assets/img/machine_learning/Decision_Tree_Regression_image_2.PNG)
  * ![1563155317500](/assets/img/machine_learning/Decision_Tree_Regression_image_3.PNG)




* Python Code ( Housing Dataset)

  ```python
  from sklearn.datasets import load_boston
  import matplotlib.pyplot as plt
  import pandas as pd
  import numpy as np
  from sklearn import metrics
  from sklearn import linear_model
  from sklearn import model_selection
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.model_selection import GridSearchCV
  import pydotplus
  from IPython.display import Image
  from sklearn.tree import export_graphviz  
  
  data = load_boston()
  df = pd.DataFrame(data.data, columns=data.feature_names)
  df.head()
  ```

![Decision Tree Regression_image_4](/assets/img/machine_learning/Decision_Tree_Regression_image_4.PNG)

```python

 
def performance_metric(y_true, y_predict):
    #mse
    error = metrics.mean_squared_error(y_true, y_predict)
    return error
 
def fit_model(data, target):
    regressor = DecisionTreeRegressor()
    param_grid = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
    scoring_fnc = metrics.make_scorer(performance_metric, False)
    reg = GridSearchCV(regressor, param_grid, scoring = scoring_fnc, cv = 3)
    reg.fit(data, target)
    return reg.best_estimator_
 
boston = load_boston()
medv = boston.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)
regression = fit_model(X_train, y_train)
#CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
client = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]
pred_house_price = regression.predict(client)[0]
 
print("Predicted value of client's home: {0:.2f}".format(pred_house_price), "(No Feature Selection)")
 

```

![Decision Tree Regression_image_5](/assets/img/machine_learning/Decision_Tree_Regression_image_5.PNG)



```python

plt.figure(figsize=(20, 5))
medv = data.target
data = df[['RM', 'LSTAT', 'PTRATIO']]
 
# i: index
for i, col in enumerate(data.columns):
    # 3 plots here hence 1, 3
    plt.subplot(1, 3, i+1)
    x = data[col]
    y = medv
    plt.plot(x, y, 'o')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title('Correlation between '+ col + ' and MEDV ')
    plt.xlabel(col)
    plt.ylabel('MEDV')
```



![Decision Tree Regression_image_6](/assets/img/machine_learning/Decision_Tree_Regression_image_6.PNG)

```python
#print(data.head()) keep 'RM', 'LSTAT', 'PTRATIO'
X_train_f, X_test_f, y_train_f, y_test_f = model_selection.train_test_split(data, medv, test_size=0.3, random_state=42)
regression_f = fit_model(X_train_f, y_train_f)
client = [[5.6090, 12.13, 20.20]]
pred_house_price = regression_f.predict(client)[0]
print("Predicted value of client's home: {0:.2f}".format(pred_house_price), "(Features Selected)")
print()
 

```

![Decision Tree Regression_image_7](/assets/img/machine_learning/Decision_Tree_Regression_image_7.PNG)



```python
y_pred = regression.predict(X_test)
plt.figure()
plt.title("Decision Tree Regressor (Model Actual vs Precited) with All Features")
plt.xlabel('TEST SET')
plt.ylabel('MEDV')
plt.plot(y_pred, 'o-', color="r", label="Predicted MEDV")
plt.plot(y_test, 'o-', color="g", label="Actual MEDV")
 
y_pred_f = regression_f.predict(X_test_f)
plt.figure()
plt.title("Decision Tree Regressor (Model Actual vs Precited) with Selected Features")
plt.xlabel('TEST SET')
plt.ylabel('MEDV')
plt.plot(y_pred_f, 'o-', color="r", label="Predicted MEDV")
plt.plot(y_test_f, 'o-', color="g", label="Actual MEDV")
```

![Decision Tree Regression_image_8](/assets/img/machine_learning/Decision_Tree_Regression_image_8.PNG)

```python
export_graphviz(regression_f, out_file ='tree.dot') 
with open("tree.dot") as f:
    dot_graph = f.read()

# remove the display(...)
pydot_graph = pydotplus.graph_from_dot_file("tree.dot")
Image(pydot_graph.create_png())
```

![Decision Tree Regression_image_9](/assets/img/machine_learning/Decision_Tree_Regression_image_9.PNG)



```python
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

regtree = DecisionTreeRegressor(max_depth=3)
regtree.fit(X, y)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_hat = regtree.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_hat, color="cornflowerblue", linewidth=2, label="predict")
plt.xlabel("x")
plt.ylabel("Y value")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

![Decision Tree Regression_image_10](/assets/img/machine_learning/Decision_Tree_Regression_image_10.PNG)



## 참고 문헌

https://medium.com/data-py-blog/decision-tree-regression-in-python-b185a3c63f2b

https://datascienceschool.net/view-notebook/16c28c8c192147bfb3d4059474209e0a/

https://dataoutpost.wordpress.com/2018/04/04/simple-feature-selection-and-decision-tree-regression-for-boston-house-price-dataset-part-1/