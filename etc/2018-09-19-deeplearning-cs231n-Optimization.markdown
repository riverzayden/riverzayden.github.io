---
layout: post
title:  "[CS231n] 강의노트 : 최적화(Optimization) - SGD"
subtitle:   "[CS231n] optimization landscapes, local search, learning rate, analytic, numerical gradient"
categories: cs231n
tags: cs231n dl optimization
comments: true
img: stanford-univ.jpg
---

## **목표** : 최적화 이해 (Optimization)

## **공부 기간**
2018.09.18.화 ~ 2018.09.19.수



## 참고 자료
- [CS231n optimization 강의노트](http://cs231n.github.io/optimization-1/)
- [CS231n linear-classify 강의노트](http://cs231n.github.io/linear-classify/)
- [CS231n optimization 강의노트 한글 번역(AI-Korea)](http://aikorea.org/cs231n/optimization-1/)
- [구글 머신러닝 단기집중과정-손실part](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent?hl=ko)




# **본문**
---

 딥러닝에 대한 기본적인 지식을 쌓고자 Standford의 CS231n 강의를 듣기로 결심했다. 우선적으로 아래 cs231n 강의노트를 공부하고, 틈틈이 이동시간에 youtube에 upload된 cs231n에 해당되는 강의를 볼 생각이다. 강의노트 위주로 정리했고 강의를 보다가 드는 의문점이나, 추가로 찾아본 것들을 같이 적었다. 이해한 사항들을 좀 더 쉽게 풀이하는 과정에서 틀린 표현이 있을 수도 있으므로 참고하면서 읽어주시길 바란다.


## **목차**
1. 소개
2. 손실함수의 시각화
3. 최적화(Optimization)
4. 경사 계산(Gradient)
5. 경사 하강(Gradient Descent)


## **1. 소개 (Introduction)**
---
**Imgae Classification** 문제에 핵심적인 두가지 요소는 다음과 같다.
- **(1) Score Function (스코어 함수)**
- **(2) Loass Function (손실 함수)**


**`Score 함수`** : 어떤 인풋을 넣었을 때 적절한 output이 나오는 함수. 우리가 흔히 생각할 수 있는 일차함수, 이차함수 등과 같이 어떤 X 값이 넣으면 거기에 대한 결과값이 나오는 함수라고 보면된다.

**`Loss function (손실함수)`** : Error를 계산한 함수. 만약 실제값이 1인데 예측값으로 3이 나왔다고 하면 error는 abs(1-3)=2 라고 할 수 있는 것처럼, 예측한 값과 실제 값과의 차이가 얼마나 되는지, 여기서는 그러한 차이를 loss로 받아들이면 된다. (이때 loss 실제값과 예측해야할 값 간의 차이외에도 model의 복잡도 등이 또 다른 loss라고 생각해줄 수 있기에 아래에서 정규화항이 loss function에 포함된다)

다음의 예를 보자
**Linear Function**이 다음과 같다면

![09182](https://user-images.githubusercontent.com/24144491/45766101-b7749f80-bc71-11e8-945f-ec2a6a72a2f5.JPG)

**SVM Loss Function**와 같은 손실함수를 표현할 수 있다.

![09181](https://user-images.githubusercontent.com/24144491/45765918-4503bf80-bc71-11e8-981a-0723fcabcd66.JPG)

(SVM Loss Function 에 대해 자세히 알고싶다면 다음 자료를 참고 : [cs231n SVM 관련 자료](http://cs231n.github.io/linear-classify/))

그래도 간단히 설명하자면,

- N은 example의 총 개수
- i는 example 중 i번째 example
- j는 class중 j번째 class
- W는 weight vector로 [~,dimension of x_i]
- y_i는 example i번째의 실제 class에 해당하는 index
- +1 의 의미는 error를 보정해주기 위한 term
- aR(W)는 정규화항 (overfitting 완화하기 위해)

위의 예에서 만약 우리가 **좋은 Score함수**가 있어서 X를 넣었는데 라벨들 중 실제 라벨(index, y_i)에 해당하는 값이 가장 높게 나왔다면, **`f(x_i;W)_j - f(x_i;W)_(y_i)`**  의 값은 항상 음수가 될 거고 +1을 고려하지 않는다면 모든 function에서 Loss 값은 0이 될 것이다. 정규화항을 잠깐 제외하고 생각해준다면 **좋은 Score함수**를 통해 나온 예측값들이 실제 라벨과 유사 혹은 실제 라벨의 인덱스 값이 가장 높게 나오므로 실제 결과를 잘 예측하는 좋은 함수라고 할 수 있다.


**좋은 Score함수**란 **좋은 Weight** vector를 갖는다는 의미고 이 **Weight**가 결국 **Loss fucntion**값을 **낮게**한다.

**나쁜 Score함수**는 **나쁜 Weight** vector를 가져 예측을 잘 못해 이 **Weight**로 계산된 Score함수 값이 잘못된 예측값을 가져오고 이는 **Loss**가 **높게** 나오게 된다.

다시말해, 우리가 **Weight**를 적절히 조절한다면 최저의 Loss를 찾을 수 있게돼 **```Optimization```** 문제를 해결할 수 있을 것이다. 그렇다면 어떻게 Loss를 최적화 시키는 Weight를 찾을 수 있을까?


## **2. 손실함수의 시각화 (Visualizing the loss function)**
___
 딥러닝의 문제를 풀때 거의 모든 경우 feature가 많은 고차원에서 정의가 된다. 따라서 시각화에 어려움이 있지만 이런 문제는 (y, w1) 혹은 (y, w1, w2)와 같이 2차원 3차원으로 몇개를 뽑아내어 시각화할 수 있다.

![3](https://user-images.githubusercontent.com/24144491/45766179-f276d300-bc71-11e8-94a3-9820579a1126.JPG)
- 3개의 1차원 점들 (x0,x1,x2)
- 정규화항 없는 손실
- y_i = i (즉, x0의 실제 label = 0, x1의 실제 라벨 = 1)
- W는 [K x D] (여기서 K는 class 수 이므로 3개, D는 dimension인데 x의 dimension이 1이므로 1)

라고 이해를 하면 총 Loss(=L)는 다음과 같다.

![4](https://user-images.githubusercontent.com/24144491/45765923-49c87380-bc71-11e8-901c-48afbbcfaffa.JPG)

이를 다음과 같이 시각화하면 다음과 같은 그래프가 그려진다고 하자.

![5](https://user-images.githubusercontent.com/24144491/45765925-4b923700-bc71-11e8-93f0-77abadec47bb.JPG)
- x축이 Weight 이고 y축이 Loss(손실)
- 왼쪽 그림은 각각의 Weight_i와 Loss_i의 관계를 그린 그래프. (w0,y), (w1,y), (w2,y)
- 왼쪽의 그림을 다 더한것이 오른쪽 그림
- 왼쪽 그림에서 W는 [1 x 1]



 사실 위에서 W를 [3 x 1] 으로 보고 보고 w0, w1, w2를 나눌 수 있게 한 거일 텐데, 위의 그림을 단순히 더한다면 사실 w0 = w1 = w2 인 하나의 weight로 표현한 거니까, 아래의 그림을 이해할 때는 W [3 x 1] 이지만 사실 W = [w, w, w] 라고 이해하는 것이 더 편할 것. 그래야 저렇게 2차원으로 그림이 나오지 않나. (강의노트만 봐서 강의에 어떤 조건을 말해주었을 수도 있을 것 같음) 어쨌거나 위의 그림은 Convex, 볼록함수 모양이고, 이런 볼록함수의 최적화는 미분을 통해서 해결할 수 있을 것처럼 보인다. (여기서 꺾이는 부분에서 미분을 할 수 없지만, subgradient가 존재하고, 이를 gradient 대신 이용한다고 한다)



## **3. 최적화(Optimization)**
___
- (1) Random Search
- (2) Random Local Search
- (3) Following the Gradient

최적화를 위해 위와 같이 3가지 방법이 있다.

- (1)은 Weight를 random 하게 초기화해 Loss를 계산하고 더 낮은 loss 를 발견하면 최적의 Weight를 수정하는 식으로 간다. 하지만 수많은 example과 수많은 차원을 초기화하고 parameter(weight) 하나하나마다 얼만큼의 범위를 초기화해주어야할지도 의문이다.

- (2)은 Weight를 random으로 초기화하고 일정 step만큼 움직이여 가면서 최적의 loss를 찾는다. 이 역시 계속 똑같은 step size만큼 움직여줘야하기 때문에 step size를 얼마로 설정해야하는지 모르며 여전히 비효율적인 면이 있다.

- (3) 만약 미분값을 이용해서 움직인다면 어떨까? 

![6](https://user-images.githubusercontent.com/24144491/45765928-4cc36400-bc71-11e8-8be0-970a69ad8cbb.png)
 loss 를 줄이려면 오른쪽으로 가야한다는 것은 눈에 보이니까 안다. 그럼 컴퓨터는 어떻게하면 이런 상황일때 오른 쪽으로 가야할까?

- (1) starting point가 pink일때, 저 점에서 미분계수는 -(negative)이다.
- (2) starting point가 blue일때, 저 점에서 미분계수는 +(positive)이다.

따라서 각 점에서 미분을 했을 때,

- (1) 그 값이 - 이면 w를 오른쪽(+)으로
- (2) 그 값이 +이면 w를 왼쪽(-)로

움직여주면 된다.

우리가 어떤 w 에서 예측된 loss를 구하고, loss function을 w에 대해서 미분한 계수에 따라서 w의 방향을 옮겨주면 되는 것이다. 이것이 경사하강법인 Gradient Descent이다.


## **4. 경사 계산 (Computing the Gradient)**
___
- numeric gradient (수치 그라디언트) - 근사
- analytic gardient (해석적 그라디언트) - 미분 이용

`numeric gradient`의 코드를 보자면 다음과 같다.
```python
def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
```
```python

# to use the generic code above we want a function that takes a single argument
# (the weights in our case) so we close over X_train and Y_train
def CIFAR10_loss_fun(W):
  return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001 # random weight vector
df = eval_numerical_gradient(CIFAR10_loss_fun, W) # get the gradient
```
```python
loss_original = CIFAR10_loss_fun(W) # the original loss
print 'original loss: %f' % (loss_original, )

# lets see the effect of multiple step sizes
for step_size_log in [-10, -9, -8, -7, -6, -5,-4,-3,-2,-1]:
  step_size = 10 ** step_size_log
  W_new = W - step_size * df # new position in the weight space
  loss_new = CIFAR10_loss_fun(W_new)
  print 'for step size %f new loss: %f' % (step_size, loss_new)

# prints:
# original loss: 2.200718
# for step size 1.000000e-10 new loss: 2.200652
# for step size 1.000000e-09 new loss: 2.200057
# for step size 1.000000e-08 new loss: 2.194116
# for step size 1.000000e-07 new loss: 2.135493
# for step size 1.000000e-06 new loss: 1.647802
# for step size 1.000000e-05 new loss: 2.844355
# for step size 1.000000e-04 new loss: 25.558142
# for step size 1.000000e-03 new loss: 254.086573
# for step size 1.000000e-02 new loss: 2539.370888
# for step size 1.000000e-01 new loss: 25392.214036
```

코드에 대한 설명은 원글에 잘 나와있으니 추가적인 설명이 필요하다면 제일 위에 참고 링크를 확인해주길 바란다.

여기서 생기는 의문점들은 다음과 같다.

- **(1) step size는 얼마나 잡아야하나? -> hyperparameter tuning**
- **(2) 효율성의 문제, 모든 example들을 다 계산해 주어야하나? -> batch size! 바로 아래에**


## **5. 경사 하강 (Gradient Descent)**
___

 경사 계산해서 업데이트 해주것도 배웠고 실전에서도 잘 쓸 수 있을지 보았더니, 수 많은 exmaple 들의 (몇 만개 부터 많게는 몇 천 만개 몇 억개 까지) 경사를 모두 계산해주어야 한다면 computation cost가 상당히 많이 들어갈 것이다. 따라서 이 중 몇 개만 뽑아 배치를 이용하는 Mini-batch Gradient Descent가 실전에서 쓰이며, 성능이 좋다는 것이 선험적으로 증명(?)되었다고 한다(rule of thumbs).

```python
# 단순한 미니배치 (minibatch) 그라디언트(gradient) 업데이트

while True:
  data_batch = sample_training_data(data, 256) # 예제 256개짜리 미니배치(mini-batch)
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # 파라미터 업데이트(parameter update)
```
* Batch size는 2의 제곱수가 입력될 때 빠르다고 한다.(32, 64, 128 같은 것)



## **6. 요약**
___

![7](https://user-images.githubusercontent.com/24144491/45765932-4f25be00-bc71-11e8-9eae-74167d2b9f88.JPG)

 딥러닝의 학습과정에 대한 설명은 아래와 같다.
 
- (1) **Data** / (xi, yi) 이미 값이 있는 x(input)들과 y(label)이 있다.
- (2) **Score함수** / 처음에는 random 값으로 W vector를 정의한다
- (3) **Data**를 **Score함수**에 넣어 결과값을 얻는다
- (4) **y(label)**과 **예측값**(score함수의 output)을 비교해 **Loss**를 계산한다.
- (5) **Loss**는 (4)번의 결과값 예측값간의 error term과 과적합을 방지하기 위한 regularization loss를 더한 값이다.
- (6) **Loss**가 더 내려갈 수 있는 지점을 찾아 **W**를 업데이트 한다. (**Optimization**)
- (7) 업데이트된 **W**를 가지고 (새로운 score함수) 다시 (3)~(6)을 반복한다.


 이 학습과정은 딥러닝의 전반적인 학습 과정의 순서이다. 이 과정에서 역시 추후에 다루어야할 문제들이 몇 가지 있는데,

- (1)번 과정에서 데이터를 어떻게 구성할지
- (2)번 과정에서 W 초기화 문제
- (5)번 과정의 Loss function 구성, Regularizaiton term을 어떻게 구성할지
- (6)번 과정의 Optimizier (Loss 최적화하기 위한 알고리즘으로 무엇을 쓸지)
- (7)번 과정의 몇번의 반복과정이 필요한지

등의 문제들이 남아있다. 간단히 각각에 대한 해결방안들은 coursera의 여러 딥러닝 강의와 CS231n 강의 및 여러 딥러닝 자료들을 훑어보면서 머릿속으로는 알고 있지만 정리된 자료가 없기에 다시 강의들과 자료들을 보면서 정리한 글을 올릴 예정이다.



# 느낀점
___

 CS231n 첫 포스팅이 끝났다. 사실 이해하는데는 많은 시간이 걸리지 않았지만 첫 포스팅에다 Markdown 문법, 여러가지 크롤링 설정, 태그 설정 등도 같이 하다보니 정신 없이 작성했다. 이것하다가 저것도 하다가 이 글 쓰다가 다른 사이트도 보다가.. 느낀점도 역시 주저리 말로 늘어 놓는 것보다 정리하면 다음에 내가 다시 글을 볼 때, 어떤 부분을 해야하는지 알 수 있으니까 정리해서 적어놓아야 겠다.

1. **Markdown** 사용법 익히고(쓰면서 체화되지 않을까..), 블로그 글 작성 시 **꿀 팁** 같은 것들 습득하기 (글 쓰는 순서, 이미지 등록 순서, pont size 등 수정해서 가독성 높이기 등등)
2. 오늘 마지막 **6. 요약부분**에서 추가적으로 포스팅하겠다고 한 **5가지** 정리해서 포스팅하기
3. 딥러닝 학습 과정과 고려해야할 사항들을 **한 눈에 시각화**해서 정리하기
4. 귀찮더라도 **계속** 글 포스팅하기

이 정도이다. 사실 엄청나게 많지만 우선적으로 고려하고 해야할 것들은 위와 같다. 앞으로 열심히 공부하고 포스팅해보자!



