---
layout: post
title: "[CS231n] 강의노트 : 신경망 Part3 (Neural Networks)"
subtitle: "cs231n 강의노트, 학습, 경사 체크, 최적화 알고리즘, 하이퍼파라미터 최적화, 앙상블"
categories: cs231n
tags: cs231n dl

img: stanford-univ.jpg
comments: true

---

## 목표

이번 주제들은 신경망에서 생각해봐야할 것들, 동적인 부분들을 다룬다. Gradient check 부터, 학습 중 주의애햐할 과정들, parameter update 알고리즘, 하이퍼파라메타 설정과 끝으로 Model Ensemble까지 살펴볼 것이다. 

## 공부기간

2019.2018.09.26.수 ~ 2018.09.28.금 
30 ~ 40분씩

## 참고자료

- [CS231n 강의노트 Neural Networks 3](http://cs231n.github.io/neural-networks-3/)
- [CS231n 강의노트 Neural Networks 3 - 한글번역 (AI-Korea)](http://aikorea.org/cs231n/neural-networks-3/)



# 본문
---



## 목차

1. 기울기 검사(Gradient check)
2. Sanity Check
3. 학습 과정에서 검사할 것들 (Babysitting the learning process)
4. 파라미터 업데이트 알고리즘 (Parameter upadates)
5. 하이퍼파라미터 최적화 (Hyperparameter Optimization)
6. 평가 (Evaluation, Model Ensembles)
7. 요약 (Summary)


## 1. 기울기 검사(Gradient check)
---

 총 11가지의 체크할 사항들이 있다.

**[1] Use the centered formula.** h가 정해져 있을 경우, h의 양쪽 둘다 검증할 수 있는 오른쪽 공식을 사용할 것을 권장한다. ( second order - 공식은 에러 텀의 O(h^2)으로 더 작아지므로)

![1](https://user-images.githubusercontent.com/24144491/46187668-ae459b80-c31f-11e8-9d9a-4cbf12dc89a9.png)

- [Order_of_approximation에 관한 Wiki자료](https://en.wikipedia.org/wiki/Order_of_approximation)

**[2] Use relative error for the comparison.** numerical gradient(수치 그라디언트 - 근사) **f′_n**이고, analytic gradient(해석적 그라디언트 - 미분) **f′a**일 때, 둘의 차이를 일정 threshold로 비교를 한다. 하지만 둘 중의 차이가 어떤 경우에서는 1이 나왔고, 어떤 경우에서는 2가 나왔다고 해서 전자가 항상 작다고 해석하면 안된다. 만약 전자의 경우 둘 값이 -1,0 이 나왔다고 하고, 두 번째 경우는 10001, 10003이 나왔다고 하면 당연히 두 번째 경우가 오차가 훨씬 작다고 말해야한다. 따라서 우리는 다음과 같은 식으로 상대 에러를 구한다.

![2](https://user-images.githubusercontent.com/24144491/46187632-953cea80-c31f-11e8-8a6a-a40ba73fe5b4.JPG)

보통 구간 [1e-4 , 1e-2] 기준으로 error가 이 구간 오른쪽에(1e-2 < error) 있으면 문제가 있다고 보고, 이 구간에 있으면 좀 불안한 정도로 본다. 이 구간 왼쪽 있으면서 objective에 kinks(미분 불가능한 점)이 있으면 괜찮다고 본다. 반면 없으면 에러가 1e-4보다 왼쪽 구간에 있어도 여전히 안심할 수 없다. 1e-7보다 작다면 좋다고 한다. 주의할 점은, 네트워크 구조가 깊어질수록 상대 에러도 높아지기때문에 약 10개 정도의 층을 가지는 신경망의 경우 에러가 1e-2정도여도 괜찮다고 한다.

**[3] Use double precision.** floating point(부동소수점)을 2개 이용하라. 우리가 실수를 나타낼때도 소수점 ~째 자리 아래로 나타낼건지, 반올림 혹은 버림 혹은 올림할 건지에 따라 그 값이 달라진다. 따라서 2개의 double floating point를 설정하면 경험적으로 에러도 줄어든다고 한다.

**[4] Stick around active range of floating point.** 다음 자료를 읽어보면 좋다. 
[“What Every Computer Scientist Should Know About Floating-Point Arithmetic”](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html). 주로 아주 작은 수로 나누거나 비교 할 때 수치적인 문제들이 발생할 수 있다. 따라서 이 경우 일시적으로 loss function을 scale up 할 수도 있다. 코드 짤때도 이런 부분을 주의하면서 작성하도록 권장하고 있다.

**[5] Kinks(미분 불가능한 점) in the objective.** 예를 들어 ReLU function에서 미분을 생각해보자. x = -1e^-6 인 경우, x < 0 이므로 analytic gradient는 0 이다. 반면, numerical gradient는 f(x + h)로 계산하게 되면 (h > -1e^-6) f(x+h) = x+h이고, (x+h) / h 의 값을 가지게 된다. 특히 ReLU, leakyReLU, Maxout 등 미분 불가능한 점들을 생기게 하는 함수들이 많은데, 이런 경우 값이 정확하지 않을 수 있다는 점을 주의해야한다.

**[6] Use only few datapoints.** datapoint를 작게 잡으면 kinks의 수도 더 적어진다. 따라서 작은 data point를 이용해서 gradient 체크를 빠르고 효율적으로 해보자.

**[7] Be careful with the step size h.** h를 작게 변경해주면 갑자기 정확도가 향상 될 수도 있다. 
이 [위키피디아 글](https://en.wikipedia.org/wiki/Numerical_differentiation)에서 보면 h에 따라 numerical gradient error가 얼만큼 변하는지 나와있는 그래프가 있는데 x 값에 따라 그 error가 달라져 h를 설정하는데 있어서 지정된 룰이 없다는 것을 시사하고 있다.

**[8] Gradcheck during a "characteristic" mode of operation.** 웨이트의 초기화된 시작점에 따라 최적화의 정도가 다르니 짧은 burn-in 시간에 학습을 끝내고 여러 학습을 돌려봐야한다.

**[9] Don't let the regularization overwhelm the data.** 너무 정규화 텀에 가중치를 크게 줘버리면 모든 weight들을 0으로 만들고자 하기에 적절한 가중치가 필요하다.

**[10] Remember to turn off dropout/augmentations.** 기울기 검사(Gradient check)를 할 때, 드롭아웃이나 데이터 augmentation등의 효과를 배제해야한다. 그렇지 않으면 이게 더 큰 numerical gardient error를 가져올 수 있다. 

**[11] Check only few dimensions.** 모든 차원을 다 검사하면 무리가 있으니, 특정 부분의 영역, 차원만 검사를 한다. (대신 그 부분의 모든 parameter에 대해 기울기를 체크해주어야한다.)


## 2. Sanity Checks Tips / Tricks
---

**[1] Look for correct loss at chance performance.** 처음에 얼만큼의 loss 값이 나올지 예측하고 비교해봐라. `예를들어` CIFAR-10의 예제에서 첫 loss는 대략 2.302라는 것을 예측할 수 있다. 10개의 클래스가 있기 때문에 찍었다고 하면 확률적으로 1/10 만큼은 맞출 수 있기에, softmax loss 값은 -ln(0.1) = 2.302 가 나오게 된다. 만약 이것보다 높게 나왔다면 초기화의 문제가 있을 거라고 생각이 된다.

**[2] Regularization strength.** 정규화율을 높이면 그만큼 loss도 높아져야 한다.

**[3] Overfit a tiny subset of data.** 전체 데이터셋으로 학습하기 전에 작은(tiny) 예제부터 학습해보길 권한다. 그래야 빨리 error 사항들을 체크해볼 수 있기 때문이다.




## 3. 학습 과정에서 검사할 것들 (Babysitting the learning process)
---

신경망을 학습하면서 체크해보아야 할 사항들에 대해 알아보자.

**[1] 손실과 학습률, 손실과 배치크기와의 관계.** 

![3](https://user-images.githubusercontent.com/24144491/46187634-953cea80-c31f-11e8-819e-f192d4034947.JPG)

왼쪽은 손실(loss)과 학습률(learing rate)과의 관계이다. 학습률이 너무 높으면 loss 가 explode 된다. 적당히 높으면 학습은 빨리 되지만(어떤 색의 curve보다 초반에 빨리 loss가 떨어짐) 특정 구간을 지나면 loss가 더 떨어지지 않는다. 또 너무 낮으면 파란색 커브처럼 천천히 학습된다. 따라서 learning rate를 조금씩 바꿔가면서 빨간색과 같은 curve를 띄게하는 학습률을 찾아야한다. 

오른쪽은 손실과 배치사이즈간의 관계인데, 배치사이즈가 1인 경우의 예시이다. 배치사이즈가 1이면 loss값이 점점 떨어지기는 하지만 Epoch당 나온 loss 값의 범위가 여전히 크다('wiggle'). 배치사이즈를 full dataset size로 해버리면 wiggle은 최소가 되겠지만 학습속도가 매우 느릴 것이다. 따라서 적당한 batch size를 잡는 것 역시 중요하다.

**[2] 학습/검증 정확도.** 

![4](https://user-images.githubusercontent.com/24144491/46187635-953cea80-c31f-11e8-8237-559cd1bc47b4.JPG)

학습 데이터만 loss가 작고(정확도가 높고) 검증 데이터에서 loss가 크다(정확도가 작다)면 이 모델은 학습데이터에 과적합되었다고 볼 수 있다. 따라서 학습데이터와 검증데이터가 같이 정확도가 올라가는지 확인해 봐야한다.

**[3] 가중치 업데이트 비율**

경험적으로 parameter 크기 : update 크기 의 비율은 1000 : 1 이 좋다고 한다. 이것보다 낮으면 learning rate가 작은것이고, 이것보다 높으면 learning rate가 꽤 높은 편이다. 

```python
# assume parameter vector W and its gradient vector dW
param_scale = np.linalg.norm(W.ravel())
update = -learning_rate*dW # simple SGD update
update_scale = np.linalg.norm(update.ravel())
W += update # the actual update
print update_scale / param_scale # want ~1e-3
```

**[4] Activation / Gradient distributions per layer**

잘못된 초기화는 학습을 느리게하거나 중단시킬 수 있다. 하지만 비교적 쉽게 이런 error를 방지할 수 있는데, 그 중 한 가지 방법은 activation/gradient의 히스토그램을 찍어보는 것이다. 만약 tanh 를 썼는데 [-1, 1] 사이의 값에서 골고루 output이 나와야하지만, 모두 0이거나, -1또는 1에 다 분포해 있으면 문제가 있다고 판단할 수 있다.

**[5] First-layer Visualizations**

첫번째 레이어의 feature를 찍어보는 것도 유용하다.

![5](https://user-images.githubusercontent.com/24144491/46187636-95d58100-c31f-11e8-8fcf-a7c14beb5aa1.JPG)



## 4. 파라미터 업데이트 알고리즘 (Parameter upadates)
---

파라미터 업데이트 알고리즘에 관한 내용을 살펴보기에 앞서, 몇 가지 사항들을 정리하고자 한다.

**Annealing the learning rate.** 위의 글에서 살펴봤듯이, learning rate를 적절히 초기화 하는 것은 중요한 사항이다. 그와 더불어 학습할 때 시간이 지남에 때라 적절히 learning rate를 조절하는 것 역시 대게 도움이 되는 경우가 많다. 직관적으로 높은 학습률이 빠른 학습을 가능하게 하고, 작은 학습률은 속도는 느린대신 loss의 최저점을 더 잘 찾을 수 있는 것을 생각해볼때, 초반에는 높은 학습률을 적용하고 학습의 후반부로 넘어가면 적절히 학습률을 줄이면 되지 않을까? 여기에 관한 3가지 종류의 적용방법이 있다.
- **Step decay.** ~번의 반복(epochs)마다 ~만큼의 learning rate를 줄이는 방법. 
- **Exponential decay.** `α = (α_0)e^(−kt)`, α_0,k 는 하이퍼파라미터 그리고 t는 반복수
- **1/t decay.**  `α = (α_0)/(1+kt)`

실전에서는 step decay를 선호하는데 최근에는 만약 계산적인 예산이 넉넉한 경우 시간이 조금 많이 걸리더라도 더 낮은 decay를 사용하도록 한다.


**Second order methods.** 이 방법은 [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)를 베이스로 한 최적화 방법인데 업데이트 하는 방법은 다음과 같다. 

![6](https://user-images.githubusercontent.com/24144491/46187637-95d58100-c31f-11e8-94c2-2a5074021eb5.JPG)

여기서 `Hf(x)`는 [Hessian Matrix](https://en.wikipedia.org/wiki/Hessian_matrix)로 두 번 미분된 값을 사용한 정방행렬이다. 두 번 미분을 하므로 미분에 대한 기울기와 볼록성 오목성 까지 판단할 수 있기 때문에 휨이 약한 방향으로는 더 공격적으로, 휨이 강한 방향으로는 더 약하게 움직일 수 있다. 따라서 굳이 hyperparameter( learning_rate )를 설정하지 않아도 된다. 하지만 실제 상황에서 적용하기에는 무리가 있다. 헤시안 매트릭스를 구할 시간과 메모리가 어마무시하고, [L-BFG](https://en.wikipedia.org/wiki/Limited-memory_BFGS)가 가장 대중적이긴 하지만 역시 전체 훈련 세트를 대상으로 계산해야한다는 단점이 있다. 이런 이유들로 지금까지 이차 근사(Second order) 방법들은 사용되지 않는다.


**first-order optimization.** 이제 본격적으로 실전에서 쓰이는 일차 근사 최적화 알고리즘들에 대해 살펴보고자 한다. **SGD, Momentum, NAG, Adagrad, RMSProp, AdaDelta, Adam** 등의 알고리즘이 있는데 [Optimization Algorithms 정리](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html) 글에서 설명을 정말 잘 해 놓았기에 지금 읽어보길 추천한다.

- **SGD**

![7](https://user-images.githubusercontent.com/24144491/46187638-95d58100-c31f-11e8-9323-b78e52cf729b.JPG)

```python
x -= learning_rate * dx #dx 는 x로 편미분한 gradient 값
```

- **Momentum**

![8](https://user-images.githubusercontent.com/24144491/46187639-95d58100-c31f-11e8-9e37-0d373ebc61e7.JPG)

```python
v = mu * v - learning_rate * dx
x += v
```
: 이때 mu는 모멘텀(운동량), 보통 0.9로 설정하고 교차 검증을 위해서는 [0.5, 0.9, 0.95, 0.99]로 설정한다. 모멘텀을 쓰면, 속도의 방향은 그라디언트들이 많이 향하는 방향으로 축적된다. 이런 모멘텀은 SGD가 Oscilation현상(지그재그로 가는 현상)을 완화시켜주기도 하고, 관성으로 인해 더 넓은 범위까지 감으로써 local minima에 빠지지 않을 수 있게 해주기도 한다. 

![9](https://user-images.githubusercontent.com/24144491/46187640-966e1780-c31f-11e8-965e-57db83997566.JPG)

- **NAG**

![11](https://user-images.githubusercontent.com/24144491/46187642-966e1780-c31f-11e8-9ec4-e8b51b249b51.JPG)
```python 
x_ahead = x + mu * v #먼저 관성으로 가고
v = mu * v -learning_rate * dx_head #간데서 미분
x += v #업데이트
```
모멘텀과의 차이를 보면 다음 그림과 같다.
![10](https://user-images.githubusercontent.com/24144491/46187641-966e1780-c31f-11e8-90af-81aa0346e117.JPG)
NAG을 이용할 경우 이동을 빨리 하고 미분을 해므로 멈춰야할 경우에 멈출 수 있다.


- **Adagrad**

![12](https://user-images.githubusercontent.com/24144491/46187644-966e1780-c31f-11e8-9c4d-e06f4190bcb4.JPG)

```python
cache += dx**2
x -= learning_rate * dx /(np.sqrt(cache)+eps)
```
cache는 그라디언트의 벡터사이즈와 동일한 사이즈이고, 그라디언트 제곲값들을 계속 쌓고있다. 이 경우 step decay를 신경써주지 않아도 learning rate를 자동적으로 조절해줄 수 있다는 장점이있다. 하지만 cache된 값들이 계속 쌓이면 쌓일 수록 그 값은 커지고 그러면 x에 업데이트되는 양이 0에 가깝기 때문에 학습이 조기 중단될 수 있다. (eps는 분모가 너무 0에 가깝지 않도록 안정화시키는 역할을 하는 변수로 1e-4~1e-8의 값이 할당된다)

- **RMSProp**

![13](https://user-images.githubusercontent.com/24144491/46187646-9706ae00-c31f-11e8-9ab3-43ec65998b99.JPG)

```python
cache = decay_rate * cache + (1-decay_rate)* dx**2
x -= learning_rate * dx / (np.sqrt(cache) +eps) 
```
Adgrad와 거의 흡사하지만, 이동평균(moving average)를 사용하여 단조감소하는 학습속도를 경감시켰다. 여기서 decay_rate 는 [0.9,0.99,0.999] 중 하나의 값을 취한다. 

- **Adam**

![14](https://user-images.githubusercontent.com/24144491/46187647-9706ae00-c31f-11e8-8ac8-e923c6847f68.JPG)

```python
m = beta1*m + (1-beta1)*dx 
v = beta2*v + (1-beta2)*(dx**2) 
x -= learning_rate * m / (np.sqrt(v) + eps)
```
이때 `eps = 1e-8, beta1 = 0.9, beta2 =  0.999`로 세팅한다. 다만, m과 v가 처음에 0으로 초기화되었을 것이라는 가정하에, m,v가 0에 bias된 것을 unbiased 되는 작업을 거친다. 

![15](https://user-images.githubusercontent.com/24144491/46187648-9706ae00-c31f-11e8-8ba4-f37e765c941d.JPG)

```python
m  = beta1 * m + (1-beta1)*dx
mt = m / (1-beta1**t)
v = beta2 * v + (1-beta2)*(dx**2)
vt = v / (1-beta2 **t)
x -= learning_rate * mt / (np.sqrt(vt) + eps)
```
<div class="fig figcenter fighighlight">
<img src="https://user-images.githubusercontent.com/24144491/46187650-979f4480-c31f-11e8-9813-ad539cdd3b4b.gif" width="48%"/> <img src="https://user-images.githubusercontent.com/24144491/46187651-979f4480-c31f-11e8-9068-99c606d546bb.gif" width="48%"/> 

  <div class="figcaption">
    **왼쪽**에서 등고선 위에서 최적화 알고리즘들의 속도를 주목하라. **오른쪽그림**은 목적함수에 안장점이 있을때 SGD의 단점을 보여준다.
  </div>
</div>




## 5. 하이퍼파라미터 최적화 (Hyperparameter Optimization)
---

이제 Hyperprameter를 Tuning, Optimize하는 방법을 알아보자. 
- learning rate
- learning rate decay
- regularization strength (L2 penality, dropout strength etc)

**[1] 범위.** learning rate는 `10 ** uniform(-6,1)`사이의 난수값으로, 보통은 0.001로 설정. dropout은 `uniform(0,1)`사이의 난수값으로, 보통은 0.2~0.5로 설정. 각 끝 경계에서 초기화되었는지는 확인해볼 것. 원래는 그 사이 난수값이면 구간의 경계값이 초기화될 일은 극히 드물기 때문이다.

**[2] 랜덤 검색 Random Search.** 

![16](https://user-images.githubusercontent.com/24144491/46187653-979f4480-c31f-11e8-99a9-37c049eb9946.JPG)

**[3] 검증시 단일 검증집합.** 적당한 크기의 검증 집합을 설정해 한 번만 검증하는 것이 더 쉽게 구현할 수 있을 것.



## 6. 평가 (Evaluation, Model Ensembles)
---


실전에서, 신경망의 성능을 끌어올릴 수 있는 좋은 방법은 여러 모형을 만들고 그 모형들의 평균값으로 에측하는 것이다. 

- **같은 모형, 다른 초기화 (Same model, different initializations).** 교차 검증으로 최고의 초모수를 결정한 다음에, 같은 초모수를 이용하되 초기값을 임의로 다양하게 여러 모형을 훈련한다. 이 접근법의 위험은, 모형의 다양성이 오직 다양한 초기값에서만 온다는 것이다.
- **교차 검증 동안 발견되는 최고의 모형들 (Top models discovered during cross-validation).** 교차 검증으로 최고의 초모수(들)를 결정한 다음에, 몇 개의 최고 모형을 선정하여 (예. 10개) 이들로 앙상블을 구축한다. 이 방법은 앙상블 내의 다양성을 증대시키나, 준-최적 모형을 포함할 수도 있는 위험이 있다. 실전에서는 이를 수행하는 게 (위보다) 쉬운 편인데, 교차 검증 뒤에 추가적인 모형의 재훈련이 필요없기 때문이다.
- **한 모형에서 다른 체크포인트들을 (Different checkpoints of a single model).** 만약 훈련이 매우 값비싸면, 어떤 사람들은 단일한 네트워크의 체크포인트들을 (이를테면 매 에폭 후) 앙상블하여 제한적인 성공을 거둔 바 있음을 기억해 두라. 명백하게 이 방법은 다양성이 떨어지지만, 실전에서는 합리적으로 잘 작동할 수 있다. 이 방법은 매우 간편하고 저렴하다는 것이 장점이다.
- **훈련 동안의 모수값들에 평균을 취하기 (Running average of parameters during training).** 훈련 동안 (시간에 따른) 웨이트 값들의 지수 하강 합(exponentially decaying sum)을 저장하는 제 2의 네트워크를 만들면 언제나 몇 퍼센트의 이득을 값싸게 취할 수 있다. 이 방식으로 당신은 최근 몇 iteration 동안의 네트워크에 평균을 취한다고 생각할 수도 있다. 마지막 몇 스텝 동안의 웨이트값들을 이렇게 “안정화” 시킴으로써 당신은 언제나 더 나은 검증 오차를 얻을 수 있다. 거친 직관으로 생각하자면, 목적함수는 볼(bowl)-모양이고 당신의 네트워크는 극값(mode) 주변을 맴돌 것이므로, 평균을 취하면 극값에 더 가까운 어딘가에 다다를 기회가 더 많아질 것이다.


## 7. 요약 (Summary)
---

신경망 훈련을 위하여

- 코드를 짜는 중간중간에 작은 배치로 그라디언트를 체크하고, 뜻하지 않게 튀어나올 위험을 인지하고 있어라.

- 손실함수 초기값이 합리적인지 판단해라.

- 학습하는 동안, 손실함수와 훈련/검증 정확도를 계속 살펴보고, (이게 좀 더 멋져 보이면) 현재 파라미터 값 대비 업데이트 값 또한 살펴보라 (대충 ~1e-3 정도 되어야 한다). 만약 ConvNet을 다루고 있다면, 첫 층의 웨이트값도 살펴보라.
- 가중치 업데이트 알고리즘으로는 SGD+Nesterov Momentum 혹은 Adam을 쓰자.
- learning rate decay. 예를 들면, 정해진 에폭 수 뒤에 (혹은 검증 정확도가 상승하다가 하강세로 꺾이면) 학습 속도를 반으로 깎아라.
- 하이퍼파라미터는 그리드 검색이 아닌 랜덤 검색으로 튜닝해라. 처음에는 생긴 범위에서 탐색하다가 (넓은 초모수 범위, 1-5 에폭 정도만 학습), 점점 촘촘하게 검색하라 (좁은 범위, 더 많은 에폭에서 학습).
- 추가적인 개선을 위하여 모델 앙상블(model ensemble)을 만들어라.


# 마무리
---

역시 공부하는 시간보다 글을 정리해서 쓰는 일이 오래 걸린다. 

- Adam Optimization 논문 Review
- 지수평균이 갖는 의미와 (like moving average) 수학적으로 증명된 효과같은 것이 있다면 찾아보자.
- optimization에 관련된 최근 trend 검색
- 시간이 되면 [이 글](http://ruder.io/deep-learning-optimization-2017/index.html#improvingadam)을 읽어보자.
- 블로그 기능, layout 손보기 ( 하루 날 잡아서 )

그래도 점차 정리하는데 쳬계가 잡혀가고 더 깔끔하게 구성하고 글을 작성하고 있는 것 같아 뿌듯하다.
