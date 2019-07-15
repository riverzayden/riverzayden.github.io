---

layout: post

title:  "[ML] bias and Variance "

subtitle:   "[ML] bias and Variance "

categories: ml

tags: ml bias variance overfitting underfitting

comments: true

img: 

---



## bias and Variance

* Hight Variance , low Bias ==> overfitting
* Low variance , High Bias ==> underfitting



* 설명

![bias_and_variance_image_1](/assets/img/machine_learning/bias_and_variance_image_1.PNG)

==>  variance가 증가하게 된다면 , 데이터의 점들의 분산은 예측력을 좀 더 떨어뜨린다. 그리고 bias가 커지게 된다면 실제값과 예측값의 오차는 커진다. 



* 그럼 어떻게?

![bias_and_variance_image_2](/assets/img/machine_learning/bias_and_variance_image_2.PNG)

​	==> 모형에 더 많은 변수를 넣게 되면 복잡성은 증가하고, variance는 늘어나고 bias는 줄게 되는 영향

​	==> 최적의 point를 찾아야 한다. ( bias의 감소가 variance의 증가와 같아지는 )

​    ==> 방법의는 모델의 복잡도를 줄이거나 , 정규화(Regularization)를 통해서 





## 참고자료 

https://brunch.co.kr/@itschloe1/11