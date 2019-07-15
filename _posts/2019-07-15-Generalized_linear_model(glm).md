---

layout: post

title:  "[ML] Generalized linear Model ( GLM ) "

subtitle:   "[ML] Generalized linear Model ( GLM ) "

categories: ml

tags: ml GLM linear regression

comments: true

img: 

---



#### Generalized linear Model ( GLM )

* 정의
  - 종속변수가 정규분포하지 않는 경우를 포함하는 선형모형의 확장 
  - family라는 인자의 따라 link함수가 달라진다.
    - 종속변수의 분포가 정규분포인 경우 Gaussian
    - 종속변수의 분포가 이항분포 경우 binomial
    - 종속변수의 분포가 포아송인 경우 Poisson
    - 종속변수의 분포가 역정규분포인 경우 inverse gaussian
    - 종속변수의 분포가 감마분포인 경우 gamma
  - 대표적모델
    - 종속변수가 0 아니면 1인 경우 : Logistic regression
    - 종속변수가 순위나 선호도와 같이 순서만 있는 데이터 : ordinal regression
    - 종속변수가 개수를 나타내는 경우 : poisson Regression



* Python 코드

```python
import statsmodels.api as sm
data = sm.datasets.scotland.load()
print(data.exog.shape)
print(data.exog[0])
## 상수항 결합을 위해 추가해주는 과정 
data.exog = sm.add_constant(data.exog)
print(data.exog.shape)
print(data.exog[0])
```

![Generalized_linear_model_image_1](/assets/img/machine_learning/Generalized_linear_model_image_1.PNG)

```python
gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma(link=sm.genmod.families.links.log))
gamma_results = gamma_model.fit()
print(gamma_results.summary())
```

![Generalized_linear_model_image_2](/assets/img/machine_learning/Generalized_linear_model_image_2.PNG)



## 참고 자료

https://www.statsmodels.org/dev/generated/statsmodels.genmod.generalized_linear_model.GLM.html

https://rstudio-pubs-static.s3.amazonaws.com/41074_62aa52bdc9ff48a2ba3fb0f468e19118.html

https://brunch.co.kr/@gimmesilver/38