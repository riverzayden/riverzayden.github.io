---

layout: post

title:  "[ML] Bagging and Boosting "

subtitle:   "[ML] Bagging and Boosting "

categories: ml

tags: ml bagging boosting

comments: true

img: machine_learning.png

---



#### Bagging과 Boosting 





1. 앙상블 기법 ( Ensemble)
   - Bagging과 Boosting이 해당 된다.
   - 동일한 학습 알고리즘을 사용하여 여러모델을 학습시킨다.
   - 서로 다른 모델을 결합하여 새로운 모델을 만들어내는 방법( Stacking ) 과 대조



2. Bagging

   - 여러번 샘플을 뽑아서 각 모델을 학습시켜 결과 집계하는 방법

   ![bagging_boosting_image_1](/assets/img/machine_learning/bagging_boosting_image_1.PNG)

   - 이렇게 하는 이유 : 알고리즘의 안정성과 정확성을 향상시키기 위해서

     - 높은 bias로 인한 언더피팅
     - 높은 variance로 인한 오버피팅

   - 오버피팅을 피할 수 잇는 가장 좋은 방법 

   - 대표적으로 Random Forest방법 

     ![bagging_boosting_image_2](/assets/img/machine_learning/bagging_boosting_image_2.PNG)

2. Boosting
   - bagging이 일반적인 모델을 만드는데 집중한 반면에, Boosting은 맞추기 어려운 문제를 맞추는데 초점을 두고 있다.
   - bagging과 동일하게 복원랜덤 샘플링을 하지만, 가중치를 부여한다는 차이
   - bagging은 병렬로 학습, boosting은 순차적으로 학습 ( 가중치 재분배를 위해서 )
   - 대표적인 모델로는 AdaBoost, XGboost, GradientBoost

![bagging_boosting_image_3](/assets/img/machine_learning/bagging_boosting_image_3.PNG)





## 참고 자료

https://swalloow.github.io/bagging-boosting