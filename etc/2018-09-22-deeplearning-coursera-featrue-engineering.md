---
layout: post
title: "[Coursera] 데이터 전처리 : Feature Engineering - Machine Learning with Tensorflow on Google Cloud Platform"
subtitle: "feature engineering, tensorflow, google cloud platform"
categories: coursera
tags: coursera dl tf
img: coursera.png
comments: true
---


## 목표

Coursera의 **"ML with Tensorflow on GCP"**의 강좌 4 - **Feature Engineering** 이해 및 정리

## **공부기간**
2018.09.22.토

## **참고자료**

[Coursera - Machine Learning with Tensorflow on Google Cloud Platform -강좌4](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp)


# **본문**
---
오늘은 **coursera**의 **Machine Learning with Tensorflow on Google Cloud Platform**의 강좌 4인 **`Feature Engineering`**에 대해 공부하고자 한다. Machine Learning 모델을 만들고 학습하기에 앞서 feature에 대한 **`preprocessing`** 과정이 매우 중요하기 때문에 강의를 꼼꼼하게 요약하고 정리할 생각이다. **Course 에 없는 내용**이나 **내 주관적인 생각**이 들어갈 수 있다는 점을 감안하고 읽어주었으면 한다.

이번 강의에서는 아래와 같은 사항들을 살펴볼 것이다.

- Raw Data 중 어떤 것들을 Feature로 선택할지
- Preprocessing 과정을 통해 Feature를 어떻게 만들지
- Feature Crosses 에 관한 내용
- TF Transform 에 관한 내용



## **목차**

1] **Introduction**
2] **Raw Data to Features**
3] **Preprocessing and Feature Creation**

-- **Preprocessing and Feature Creation**
-- Apache Beam and Cloud Dataflow
-- Preprocessing with Cloud Dataprep

4] **Feature Crosses**
5] **TF Transform**
6] **Summary**


## **1. Introduction**
---

ML Model의 정확도를 향상시키기 위해는 2가지 방법이 있다.

- (1) Feature Engineering
- (2) Art and Science of ML

**`(1)`** 은 Model **구성 전** data를 잘 가공하는 과정이고,

**`(2)`** 은 Model **구성과 학습할 때** Hyperparameter Tuning / Regularization / Neural Networks 구조/ Embedding 에 관한 문제를 다룬다.

오늘은 여기서 **`(1)`**에 대한 내용을 살펴볼 것이다.

큰 흐름은 
- **Data** -> **Feature** 변환하는 문제
- **Feature** -> **Feature**(add, delete, edit, create)
: feature를 추가하거나, 수정하거나, 삭제하거나 새로운 feature를 만드는 것에 대한 문제를 살펴볼 것 같다.


## **2. Raw Data to Features**
---

이번 절에서는 어떻게 raw data로부터 feature들을 조작하고 생산하는가에 대한 문제를 다룬다. 이번 절의 목표는 다음과 같다.

![3](https://user-images.githubusercontent.com/24144491/45911712-27da1700-be52-11e8-9c14-435a3d436eff.JPG)

- (1) raw data -> feature로
- (2) good feature 선택
- (3) Represent feature (feature 표현)



### **(1) Raw data -> Feature**

![2](https://user-images.githubusercontent.com/24144491/45911711-27da1700-be52-11e8-99b9-edab0dde40ff.JPG)

- **Numerical feature vector로 매핑하기**
: 우리가 구성할 모델은 대부분 적절한 계산을 통해 일정한 값을 도출해내는 여러가지의 함수들로 혼합되어있다. 따라서 이 함수가 **계산**되어야 하므로 **numerical(숫자로 된)** 변수들로 **mapping** 시켜주어야한다.

### **(2) Good Feature 선택**

![4](https://user-images.githubusercontent.com/24144491/45911713-2872ad80-be52-11e8-8592-2cc4b6c1961d.JPG)

**Good feature**는 

- [1] 구하려는 값과 관계가 있어야한다.
- [2] 예측할 당시 알 수 있는 feature여야 한다.
- [3] 규모(magnitude)가 의미를 갖는 numeric 이어야한다.
- [4] 충분한 example수가 있는 feature여야 한다.
- [5] 인간의 직관과 통찰으로 선택할 수도 있다.

#### **2-1. 구하려는 값과 관계**

 이 feature와 우리가 세운 가설간의 인과적인 관계가 있는지부터 생각해보면 된다. 왜 이 feature가 변할 때 우리가 예측하려고하는 값이 변할까를 물었을 때, 적절한 관계가 보이면 당연히 그 값을 예측하는 좋은 feature가 될 것이다. 하지만 관계가 보이지 않는다고해서 data를 임의로 버릴 수 없다. 그때는 각 데이터간의 상관관계(correlations)가 있는지 살펴본다음, 상관관계가 눈에 보인다고 판단 되면 그 feature를 일단 선택하자. 이 때, feature들과 예측할 label(y)값의 관계를 시각화하는 과정이 필요할 것이고, 이에 대해서는 추후에 kaggle에 대한 글을 포스팅할 때 확인해 볼 수 있을 것이다. 
 ( kaggle에 들어가보자. 아무 project의 open kernel을 들어가면 거의 모든 사람들이 ML 모델을 구성하기 전 data 시각화를 하는 과정을 거치는 것을 볼 수 있다. 이때 데이터 시각화는 cor을 관찰하기 위함도 있지만 다른 몇몇의 과정들을 위해서이기도 하다 )

 다음의 **`예시`**를 보자

![6](https://user-images.githubusercontent.com/24144491/45911715-2872ad80-be52-11e8-96b3-ec89461c6e5d.png)

좋은 경주마를 예측하기 위해 다음과 같은 feature들이 있다고 하면 어떤 feature들을 선택해야할까?

`A`와 `B`는 말이 잘 달리는지의 경주 실력과 확연한 관계가 있는 것으로 보인다.
반면, Eye color는 어떨까? 언뜻 보기에 관계가 없어보이지만 만약 눈의 색으로 눈에 이상이 없는지 있는지 알 수 있다면, 그 말이 경주를 할 때 앞을 잘 볼지 못 볼지 예측할 수 있는 것이고 그런 지표는 말의 경주실력과 연관이 있을 것이다.

`D `같은 경우 등번호인데, 사실 어떤 등번호를 달고 나왔는지는 경주실력과 관계가 없을 것이다. 따라서 D같은 특성은 좋은 경주마를 예측하는데 필요없는 변수로 보이므로 이 data들은 굳이 feature로 변환해주지 않아도 될 것이다.
 
#### **2-2. 예측할 당시 알 수 있는 feature여야 한다.**

당연히 예측을 할 때 feature를 알 수 없다면 거기에 해당한 input을 넣을 수 없을테니 예측할 시기에 우리가 수집할 수 있는 feature여야 한다.

#### **2-3. 규모(magnitude)가 의미를 갖는 numeric feature**

 다음의 **`Quiz`**를 풀어보자.

>**Which of these features are numeric (or could be in a useful form)?**

>**`Objective`**: Predict total number of customers who will use a certain discount coupon

>(1) Percent value of the discount (e.g. 10% off, 20% off, etc.)
(2) Size of the coupon (e.g. 4 cm2, 24 cm2, 48 cm2, etc.)
(3) Font an advertisement is in (Arial, Times New Roman, etc.)
(4) Color of coupon (red, black, blue, etc.)
(5) Item category (1 for dairy, 2 for deli, 3 for canned goods, etc.)

**`(1)번`**의 경우는 할인율이 높으면 높을수록 사용할 확률은 높아질 것이다. 할인이 더 많이 되니 사용하게 될 동기가 더 높아질 거니까. 따라서 이 경우는 할인율이 높으면 높을수록 사용률도 높아지니 할인율의 규모, 크기는 어떤 의미를 가진다.

반면,
**`(2)번`**의 경우를 보자. size가 크면 클수록 사람들이 더 사용을 많이 할까? 혹은 크면 클수록 덜 사용할까? 만약 그렇다면 크기는 의미를 갖는 변수가 되겠지만 잘 생각해보면 특정 디자인이나 size에 민감하여 어떤 크기가 사용률과 제일 관계있을지 모른다. 따라서 이런 변수들은 numeric보단 categorical 변수로 넣는 것이 더 합리적인 선택이다.

나머지 경우는 스스로 생각해보자. (답은 1번 뿐이다)

**```여기서 잠깐```**

**`Q1.`** 어떤 경우일때 **numerical**이고 어떤 경우일때 **categorical**으로 넣는가?
**`A1.`** 앞서 살펴본 것처럼 각 **feature의 크기가 의미**를 갖는다면 **numerical**, 그렇지 않다면 **categorical** 변수로 취급해주면 된다.

**`Q2.`** 왜 굳이 **numerical**과 **categorical**을 나누는 것일까?

**`A2.`** 이 해석에 있어선 다소 나의 직관적인 견해이긴 하지만, **categorical** 변수를 **numerical** 변수로 취급했을때는 다음과 같은 문제가 발생할 것 같다.
- numerical feature들은 1개의 weight vector이 붙게된다.
- categorical feature들은 각 category 마다 1개씩의 weight vector들이 붙게된다.
- 보통 ML 모델을 학습할 때 Loss를 최적화(optimization)하기 위해 미분값을 이용한다.
- 크기에 의미가 없는 변수들은 각 변수간의 y예측 값을 그렸을때 들쭉 날쭉이다. 없는 종류의 값들에 대해서는 y값들이 없다.
- categorical 변수들은 여러가지의 복합적인 함수형태로 나타내야되며 나타내더라도 없는 값들에 대해서 선형적으로 혹은 연결되어 있다고 보장하기 어렵다. (미분 불가 함수)
- 그러므로 학습시 미분을 사용하지 못하고, 행여 사용하더라도 Loss 곡선이 여러 협곡이 있는 산처럼 형성 될 것이므로 global minima에 도달하지 못할 가능성이 크다.


#### **2-4. 충분한 Example수**

당연히 그 feature의 값을 가진 예가 별로 없다면 모델을 학습할때 적절한 모델이 나오기 힘들다.(그 feature가 label을 에측하는데 정말 유의미한 상관관계가 존재하지 않는 이상) 따라서, 충분한 example이 있는 feature를 선택하자.

#### **2-5. 인간의 통찰력**

인간의 놀라운 통찰력을 발휘해보자! (너무나 당연한 것들은 우리가 쉽게 넣고 빼고 할 수 있으므로.)



### **(3) Represent feature**

다음의 예를 보자. 어떤 ice cream 가게에서 소비자가 얼마나 만족했는지를 알고싶다고 하자. 소비자의 만족도를 예측하기 위해 다음과 같은 data들이 있다.

![7](https://user-images.githubusercontent.com/24144491/45911716-2872ad80-be52-11e8-8653-5a4a0b3223be.JPG)

이런 json data 가 있으면 다음과 같인 feature들을 만들 수 있다.

![8](https://user-images.githubusercontent.com/24144491/45911717-290b4400-be52-11e8-9fa3-a784b4201546.JPG)

이미 numeric인 변수들을 encode하는 것은 어렵지 않을 것이다.

![9](https://user-images.githubusercontent.com/24144491/45911718-290b4400-be52-11e8-9737-cf8e1fc29d9e.JPG)

위의 두 변수들은 이미 숫자에다 규모가 의미를 갖는 numerical feature들이다. 그래서 쉽게 encode해서 그대로 쓰면 된다.

반면, 다음과 같은 input은?

![10](https://user-images.githubusercontent.com/24144491/45911719-290b4400-be52-11e8-9aae-e9ba5eed108a.JPG)

이런 **categorical** 변수들은 **one-hot encoding**을 해주자.

![11](https://user-images.githubusercontent.com/24144491/45911720-29a3da80-be52-11e8-9bdc-52a4e8556c36.JPG)

코드는

![12](https://user-images.githubusercontent.com/24144491/45911721-29a3da80-be52-11e8-904c-cb98190f90ae.JPG)

이렇게 해주면, keys를 ID로해서 column name이 employee ID로 된 sparse column이 만들어진다. 이렇게 해야 나중에 학습이 끝나고 예측된 값들을 볼때 어떤 employee가 영향을 주고 안주는지를 알 수 있으니까.

실전에 대한 팁은 다음과 같다.

![13](https://user-images.githubusercontent.com/24144491/45911722-29a3da80-be52-11e8-9dd3-ec86ef8cd1f7.JPG)

이미 key를 알고 있을때, index로 매겨져 있을 때 그리고 둘 다 없을 때이다.
둘 다 없을때는 그냥 어림짐작으로 bucketsize를 크게 잡아서 넣으면 해결.

rating과 같은 숫자들은 규모가 의미를 가지긴 하지만 각 rating 간의 규모의 차이가 일정한 크기가 아니기 때문에 이런 변수들은 categorical 변수로 넣어도 좋다.

![14](https://user-images.githubusercontent.com/24144491/45911723-29a3da80-be52-11e8-8e65-ed66a1abe802.JPG)

마지막으로 missing data의 처리는 어떻게 해야할까? 이부분은 technical하게 rating 표시했는지 안했는지 체크하는 칸을 만들라고 하는데, 소비자가 rating을 안했건, 넣을때 누락되건 간에 결국 missing data는 이미 잃어버린 data이다. 따라서 이런 data는 특별히 다른 방법으로 처리를 해주어야할 것 같은데 이 부분에 대해서 추후의 공부가 필요할 것 같다. (missing data처리 관련, kaggle에서는 그냥 0값을 넣거나 avergae를 넣던데, missing data가 많으면 문제가 되겠지? 이 때 어떻게 처리하는지는 한 번 찾아보자. 아래의 참고하면 좋은 자료의 링크 참고하길 바란다)

>### **`ML vs Statistics`**
>여기서는 ML model 구성시 data가 없는 상황과 그렇지 않은 상황에 대해 별도의 모델을 작성하는 것이 좋다고 한다. 
- **통계**는 제한된 데이터로 돌리고, **ML**은 충분히 많은 데이터로 작동
- **통계**는 outlier 버리지만, **ML**은 충분한 데이터를 찾았으므로 아웃라이어라도 같이 돌릴것(?)


### **참고하면 좋은 자료**
- [기본적인 데이터 종류](http://blog.heartcount.io/dd)
- [Missing data 처리 전략](http://sacko.tistory.com/)
(SMOTE, KNN 같은 방법을 사용해서 근사한 instance의 값으로 채우기가 가장 과학적인 방법인 것 같다.)
- [ML vs Statistics 1](https://www.svds.com/machine-learning-vs-statistics/)
- [ML vs Statistics 2](https://www.educba.com/machine-learning-vs-statistics/)
- [2절 관련 실습 자료-github](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/04_features/a_features.ipynb)


## **3. Preprocessing and Feature Creation**
---

3번에서는 크게 2가지 부분

- Preprocessing and Feature creation 부분
- Apache beam and Cloud 를 사용한 preprocessing

에 대해 배운다.


## -- **Preprocessing and Feature Creation**

데이터 전처리라고 하는 부분에서 어떤 작업들을 할까?

- **Scale 조정**
- **Categorical 변수 처리**

![16](https://user-images.githubusercontent.com/24144491/45911725-2a3c7100-be52-11e8-925b-711c848f0c8c.JPG)


>참고로 Scale 조정은 (1) Standard Scaler, (2) Robust Scaler, (3) Minmax Scaler, (4) Normalizer 4가지 방법이 있는데 [Scikit-Learn의 전처리 기능](https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/)을 통해 각 Scaler가 어떻게 구성되어있는지 참고하길 바란다.


이런 과정들을 

- (1) Big Query
- (2) Apache Beam
- (3) Tensorflow

로 어떻게 구현하는지 살펴볼 것이다.

![17](https://user-images.githubusercontent.com/24144491/45911726-2a3c7100-be52-11e8-84cd-7b2a22d41d0b.JPG)

여기서 **Compute time-windowed statistics** 부분을 보자. 예를 들어, 만약 우리가 지난시간동안 웹사이트에서 판매된 상품의 수의 평균을 알고싶다면, 이런 time-windowed features를 위해 Beam의 batch와 streaming 데이터 파이프라인을 사용할것이다.
> **For example**, you can compute a time window average. Like the average number of bicycles at a traffic intersection over the past hour. However, you will have to run your prediction code also within a pipeline so that you can get the average number of bicycles at a traffic intersection over the past hour. So, this is good for examples like time window averages where you need a pipeline in any case. 


다음은 `Tensorflow` 혹은 `Beam`으로 구현해볼 수 있는 것들이다.

![18](https://user-images.githubusercontent.com/24144491/45911727-2ad50780-be52-11e8-91ad-81b4f784f2c1.JPG)

**`Q.` 그럼 왜 Beam을 사용해야할까?**

**`A.`** **`Beam`**은 training과 serving of model에서 **동일한 코드**를 사용할 수 있다.

일단 **`BigQuery`**는 상당히 확장성이있고(scalable), 매우 빠르고, Google Cloud에서 관리할 수 있다. **`예를 들어`**, 100억 건의 텍스트를 가진 데이터셋을 전처리한다고 하자. 이 데이터 레코드 중 일부는 **bogus data**(가짜 데이터)가 발생할 수 있다. 이런 **bogus** data를 where문을 통해 걸러내고 feature를 select 할 수 있다.

![19](https://user-images.githubusercontent.com/24144491/45911728-2ad50780-be52-11e8-82d0-2818ce332cc8.JPG)


이런 과정은 **`tensorflow`**를 통해서도 충분히 똑같이 해줄 수 있다. 다음은 그 예제 코드이다.

![20](https://user-images.githubusercontent.com/24144491/45911729-2ad50780-be52-11e8-86c3-5d7e4fd0d41f.JPG)


이렇게 feature를 추가하는 함수를 만들어 처리할 수도 있고

![21](https://user-images.githubusercontent.com/24144491/45911730-2b6d9e00-be52-11e8-9219-9f7324644653.JPG)


구현해야할 함수들이 **`tensorflow`** API에 있다면 Thanks a lot! 이다. 아래코드는 위도를 특정 bucket을 잘라 categorical 변수처럼 만드는 코드다.

>[이와 관련한 버킷화의 좋은 코드 예제 - 종현님 블로그](http://excelsior-cjh.tistory.com/175)

![22](https://user-images.githubusercontent.com/24144491/45911731-2b6d9e00-be52-11e8-8275-5b57ed680d79.JPG)


하지만 이런 전처리 코드를 빅쿼리와 텐서플로으에서 유지관리하는 것은 꽤 복잡하고 힘들다고한다. 하지만 Beam은 training과 serving of model에서 동일한 코드를 사용할 수 있다. 그런데 또 Beam을 사용하면 Tensorflow의 편리한 method들을 사용할 수 없다. 사용자가 전처리 코드를 직접 구현해야한다.

>(이 부분에 대해서는 강의의 부분을 그대로 번역하긴 했지만, tensorflow의 serving과정에서 역시 동일한 코드를 쓸 수 있지 않을까 생각이 든다. 이 부분은 추후에 더 살펴봐야겠다)



**정리된 내용**은 다음과 같다.

![23](https://user-images.githubusercontent.com/24144491/45911732-2b6d9e00-be52-11e8-82ac-004f13560a7d.JPG)

>아래 부분은 내용도 많고 각 부분에 대해 더 공부를 하고 따로 포스팅할 예정이다.
>
-- **Apache Beam and Cloud Dataflow**
>
-- **Preprocessing with Cloud Dataprep**


## **4. Feature Crosses**
---

**Feature cross**는 두 개 이상의 특성을 곱하여(교차하여) 구성되는 **합성 특성**이다.

구글 [플레이 그라운드](http://playground.tensorflow.org/)에서 몇 가지 실험을 해보았다.

- Neural Network만 사용해서 학습
![24](https://user-images.githubusercontent.com/24144491/45911733-2c063480-be52-11e8-9f42-e270d242c48e.JPG)

- Feature Cross도 넣고 학습
![26](https://user-images.githubusercontent.com/24144491/45911734-2c063480-be52-11e8-85c7-9a52cadef20e.JPG)

- Neural Network  vs  Neural Network + Feature Cross
![27](https://user-images.githubusercontent.com/24144491/45911735-2c063480-be52-11e8-811f-8aab85e2ef63.png)

지금까지 **Neural Net**을 **Layer**와 **Neuron**의 수를 적절히 조정하면 합성특성을 나타내는 것과 동일한 효과를 내는 어떤 **Neuron**이 만들어지지 않을까 싶었는데, ( x1 * x2 를 w1x1 + w2x2 로 나타내려면,결국 w1, w2가 각각 (1/2 x x2), (1/2 x x1) 이면 `x1 * x2 = w1x1 + w2x2`이므로 ) 

실제로 그런 **weight**를 적절히 찾기 힘들었을 수도 있고 **Neural Net**만으로 이루어진 모델은 생각보다 **과적합**되는 경우가 많았다. **`Neural Net + 특성교차`**를 넣은 모델들이 학습속도도 빠르고 적은 뉴런으로 **좋은 성능을 내는 모델**이 나왔다. 어떤 **특성들의 교차**를 통해 유의미한 새로운 특성을 만들 수 있다면 더 좋은 모델을 만들 수 있을 것이다.

특히 이런 특성교차는 **categorical** 변수들의 **교차**에 의해서 **더 좋은 feature**들을 생성해 낼 수 있다.


다음과 같은 **`예시`**를 생각해보자.

![28](https://user-images.githubusercontent.com/24144491/45911736-2c9ecb00-be52-11e8-8b30-de81290e35b6.JPG)

특히 이런 시간데이터는 숫자로 표현되지만 규모가 의미를 가지는 numeric 변수가 아니다.
**`예를들어`** 교통체증이 심할지 심하지 않을지 예측하는 모델을 만든다고 해보자. 그때 단순히 시간과 요일을 one-hot encoding해서 넣는 것이 좋을까? 우리는 흔히 퇴근시간, 출근시간에 교통체증이 심하다는 것을 알고 있다. 따라서 퇴근시간, 출근시간대를 특정한 categorical로 만들면 단순히 시간을 넣는 것 보다 훨씬 적은 vector로 표현할 수 있고 예측 성능은 훨씬 개선될 것이다. 즉, 그렇게 나눈 시간대가 **출근시간** 혹은 **퇴근시간**의 의미를 가지게 되면서 의미를 가지게 되는 feature가 되는것이다. 하지만 여기서 끝이 아니다. 보통 출근과 퇴근은 **평일**에 하게 된다. 따라서 이런 특성을 요일과 함께 나타내면, **[출근시간대이고 평일], [출근시간대이고 주말]** 두 가지의 종류의 feature들 중 앞의 특성교차는 교통체증을 알 수 있는 유의미한 특성이 되게 될 것이고, 이런 변수를 만들어서 넣는다면 더 **좋은 모델**을 만들 수 있게된다. 이 예시 외에도 특정 **categorical** 변수들이 서로 교차해 어떤 의미를 만들어내는 새로운 feature가 될 수 있는 경우는 많으니 한 번 생각해보자.

> **[주의할 점]** **항상 의미가 있는 특성교차를 만들어야한다.** (아무거나 막 교차해서 만들면 오히려 좋지 못한 모델을 만들어 냄, 과적합, 학습속도 더 느려짐 등의 이유로인해)

#### **참고하면 좋은 자료**
[구글 머신러닝 단기집중과정-onehot encoding 특성교차](https://developers.google.com/machine-learning/crash-course/feature-crosses/crossing-one-hot-vectors?hl=ko)

## feature cross 구현
---

![29](https://user-images.githubusercontent.com/24144491/45911737-2c9ecb00-be52-11e8-9419-481ed521473a.JPG)

![30](https://user-images.githubusercontent.com/24144491/45911738-2c9ecb00-be52-11e8-8b15-a0556ffdc116.JPG)

**`crossed_column`** 

( 

 [ 크로스할 **categorical** 혹은 **categoricalized** columns(2개 이상 올 수 있음) ],
 
 [ bucket을 **hash할 값**]
 
 )
 
![31](https://user-images.githubusercontent.com/24144491/45911739-2c9ecb00-be52-11e8-8f1b-e3b52e503746.JPG)

> **Bucket Hash**의 **`예`**를 생각해보자. 과일은 딸기, 수박 2가지 종류가 있고 채소는 양배추, 시금치, 옥수수 3가지 종류가 있다고 하자. 그럼 **[과일, 채소]** 했을 때 생겨나는 조합은 [딸기, 양배추], [딸기, 시금치], [딸기, 옥수수], [수박, 양배추], [수박, 시금치], [수박, 옥수수]로 총 **6가지**이다. 이렇게 cross 된 값들은 새로운 종류로 본다면 총 6종류가 생기고 이 때 one-hot encoding을 하면 [수박, 양배추]의 경우는 [0,0,0,1,0,0] 으로 encoding이 된다. 

> **하지만**, bucket(이런 category들)을 3으로 **`hash`**한다고 했을때, 각 종류의 인덱스를 3으로 나눈 나머지는 [0,1,2,0,1,2]가 될 것이고, 그렇게 된다면 [딸기,양배추]와 [수박,양배추]는 같은 종류로 인식, [딸기, 옥수수]와 [수박,옥수수]는 같은 종류로 인식한다. 그렇게 one-hot encoding된 결과는 [0,0,0] 3개의 종류로 mapping이 될 것이고, [수박,시금치] = [딸기, 시금치] = [0,1,0]로 encoding이 될 수 있다.


**`Q`.**  만약 **`hash`** 값을 너무 작게 잡거나 너무 많이 잡는다면?

![32](https://user-images.githubusercontent.com/24144491/45911740-2d376180-be52-11e8-852a-5385e6d14463.JPG)
- 작게 잡으면 그만큼 충돌(중복)된 값이 많이 일어지만 차원 축소가 됨
- 크게 잡으면 중복은 일어나지 않지만 sparse하게 encoding되어 차원이 증가됨 

![33](https://user-images.githubusercontent.com/24144491/45911800-25c48800-be53-11e8-9680-f81208df22da.JPG)

- **첫번째 층** : 하루 중 시간 / 요일의 one-hot encoding
- **두번째 층** : 하루 중 시간 x 요일  cross feature의 one-hot encoding (이 과정에서 적절한 hash값을 이용해 차원을 축소될 수 있음)
- **두번째 층과 세번재 층**의 **연결선** : 각 선마다 고유의 weight들
- **세번째 층** : 노란색 노드(자동차의 교통량) / 초록색 노드(보행자와 자전거의 교통량)


![34](https://user-images.githubusercontent.com/24144491/45911801-265d1e80-be53-11e8-8d6a-22ed04d7f342.JPG)

**`예를 들어`**, 
- `아침 8시 화요일과 아침 9시 수요일`은 보행자/자전거와 자동차의 교통량 둘다 많다.
- `오전 11시 화요일과 오후 2시 수요일`은 보행자/자전거의 교통량은 적지만 자동차의 교통량은 많다.
- `새벽 2시 화요일과 새벽 2시 수요일`은 보행자/자전거와 자동차의 교통량 둘다 적다.
이런 교통량이 공통적인 특정 시간대와 요일을 묶으면 차원을 더욱 축소할 수 있으면서 유의미한 `cross-feature`을 만들 수 있다. 

그럼, 이것을 `tensorflow`로 구현본 코드는 다음과 같다.

![35](https://user-images.githubusercontent.com/24144491/45911799-25c48800-be53-11e8-9d7b-36ab65e7f516.JPG)

>[이와 관련한 embedding의 좋은 코드 예제 - 종현님 블로그](http://excelsior-cjh.tistory.com/175)


## **5. TF Transform**
---

**`tf.transform`**을 이용하여 **preprocessing**하는 과정을 소개하는 절인데, 이 부분 역시 **`tensorflow`** 에 관한 글로 따로 포스팅할 예정이다.

## **6. Summary**
---

 이번 강의에서는 아래와 같은 것들을 살펴보았다.
 
- **feature engineering** 의 **필요성**
- **Raw Data -> Feature** 
- Feature **Select**
- **Numerical** Feature vs **Categorical** Feature
- Feature **Scale** 조정
- **One-hot encoding**
- **Feature-Cross**
- Feature-Cross **hash**


# **마무리**
---

이번 역시 설명해야할 양들이 많다보니 글을 작성하는 시간이 오래 걸렸다. 빠르면서도 간결하고 핵심적인 내용들을 적을 수 있도록 노력하는 중이다. 글의 내용이 너무 많을 때는 한 번에 다 담으려 하지말고 나눠서 작성해야겠다. 

- 포스팅 양이 많으면 나눠서 작성하기
- Beam, BigQuery, Tensorflow Transform 관련 추가 포스팅
- coursera 강의 들으면서 글 정리하기
- 핸드폰으로도 메모하면서 천천히 강의듣기 (주로 폰으로 강의를 들으니까)
- **중요한 문장**, **중요한 단어**, `코드 내용`, `예시`, `번호`, `특정 함수`, `Q&A`
>**추가설명** 이 규칙으로 **bold**, `code`, > indent 를 사용하자.
