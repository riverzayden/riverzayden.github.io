#### Random Forest Classification



* 정의
  * 랜덤 포래스트는 앙상블 기법 중 하나이다.
  * 학습데이터에서 랜덤하게 선택된 서브데이터로 디시전트리를 한 후에 종합한다. 



* Python Example

```python
import pandas as pd
# list for column headers
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# open file with pd.read_csv
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", names=names)
print(df.shape)
# print head of data set
df.head()
```

![random_forest_classification_image_1](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\random_forest_classification_image_1.PNG)

```python
## 당뇨병 여부 예측
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
X = df.drop ( 'class', axis = 1) 
y = df [ 'class']
 
# train-test-split을 구현한다. 
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.33, random_state = 66)

# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# predictions
rfc_predict = rfc.predict(X_test)
```

![random_forest_classification_image_2](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\random_forest_classification_image_2.PNG)

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
print()

accuracy = accuracy_score(y_test, rfc_predict)

print(f'Mean accuracy score: {accuracy:.3}')
```

![random_forest_classification_image_3](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\random_forest_classification_image_3.PNG)

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(X_train, y_train)
# print results
print(rfc_random.best_params_)
```

![random_forest_classification_image_4](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\random_forest_classification_image_4.PNG)

```python
rfc = RandomForestClassifier(n_estimators=200, max_depth=220, max_features='auto')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
accuracy = accuracy_score(y_test, rfc_predict)

print(f'Mean accuracy score: {accuracy:.3}')
```

![random_forest_classification_image_5](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\random_forest_classification_image_5.PNG)

## 참고 자료 

https://medium.com/@hjhuney/implementing-a-random-forest-classification-model-in-python-583891c99652