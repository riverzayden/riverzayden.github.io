#### KNN( K-Nearest Neightbors ) Classification



* 정의
  * KNN regression과 같지만 차이점은 범주형라벨을 이용하여 분류를 한다는 점 
  * 참조 : KNN regression ( 링크달기 )
* Python Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# list for column headers
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# open file with pd.read_csv
dataset = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", names=names)
print(dataset.shape)
# print head of data set
dataset.head()
```

![knn_classification_image_1](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\knn_classification_image_1.PNG)

```python

# split data into X and y
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]
# split data into train and test sets
seed = 1
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
predicted_values = neigh.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predicted_values)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

![knn_classification_image_2](D:\HBEE회사\python자료\정리본\md_image\2019-07-16\knn_classification_image_2.PNG)





## 참고 자료

https://medium.com/machine-learning-101/k-nearest-neighbors-classifier-1c1ff404d265