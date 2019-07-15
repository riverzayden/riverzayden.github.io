---
layout: post
title: "[CS231n] 강의노트 : 역전파(Backpropagation) 이해"
subtitle: "backpropagation, chain rule, patterns in gardient flow"
categories: cs231n
tags: cs231n dl
img: stanford-univ.jpg
comments: true
---

## **목표**
역전파 (Backpropagation)에 대한 직관적인 이해를 바탕으로 backprop의 과정과 세부요소들을 살펴보는 것

## **공부기간**
2018.09.20.목 ~ 2018.09.21.금

## **참고자료**
- [CS231n 강의노트 Backpropagation](http://cs231n.github.io/optimization-2/#intro)
- [CS231n 강의노트 Backpropagation-한글번역](http://aikorea.org/cs231n/optimization-2/#intro)


# **본문**
---
오늘은 **CS231n**의 Course Note 중 **Module 1: Neural Networks**의 4번째 순서에 있는 **Backpropagation**에 대해서 공부해보고자 한다. 참고로 이번 강의노트를 이해하기 위해서는 미분에 대한 개념과 행렬에 대한 곱셈 과정을 알아야 이해하기 쉽다. 간단한 설명도 추가할거지만 이 글로 이해가 잘 되지 않는다면 **미분**, **chain rule**, **행렬의 연산** 등에 대해 추가적으로 찾아보길 적극 권장한다. (나중에 여유가 된다면 이런 부분에 대해서도 포스팅하거나 관련 개념을 쉽게 이해할 수 있는 링크를 추가할 계획이다)

## **목차**
1. 소개
2. 그라디언트에 대한 간단한 표현과 이해
3. 복합 표현식(Compound Expression), 연쇄 법칙(Chain rule), Backpropagation
4. Backpropagation에 대한 직관적인 이해
5. 모듈성 : 시그모이드 예제
6. Backprop 실제 : 단계별 계산
7. 역방향 흐름의 패턴
8. 벡터 기반의 그리라디언트 계산


## **1. 소개**
---
앞선 [Optimization 글](https://taeu.github.io/dl/2018/09/19/deeplearning-cs231n-Optimization/)에서 딥러닝의 학습과정을 살펴보았다. **Loss**를 최적화하기 위해 **W**로 **미분한 값(Gradient)**을 이용했다. 복잡한 네트워크에서 역시 Loss를 최적화하기 위해 W를 미분하고 그 값을 이용해 W를 업데이트 시켜주면 된다. 하지만 W가 여러 네트워크를 거쳐 Loss로 가기때문에 단순히 바로 Loss를 W로 미분해줄 수 없다. 그래서 여기서 필요한 개념이 **Chain rule**이고, 이 chain rule을 이용해 미분한 값들을 연쇄적으로 곱해주어 gradient를 얻을 수 있는데 이러한 과정이 **Backpropagation**이다. 예시들과 함께 이 내용들을 차근차근 들여다보자.


## **2. 그라디언트에 대한 간단한 표현과 이해**
---
간단한 함수 f(x,y) = xy 를 보자

![1](https://user-images.githubusercontent.com/24144491/45867887-6a064880-bdbf-11e8-98aa-4e951a14fa5c.JPG)

여기서 먼저 **용어 정리**
- 편미분 : partial differential equation로 어떤 부분적인 독립변수로 함수값을 미분하는 것
- ∂ (델타) : 편미분 기호
- ∂f / ∂x : x라는 변수로 f를 편미분한다는 뜻

그렇다면 
f를 x로 미분하면 y가 되고,
f를 y로 미분하면 x가 된다.

**편미분할 때**, 편미분하는 독립변수빼고는 다 상수값을 취급해주면 이해하기 쉽다. 예를 들어, z = xy일때 z를 x로 미분한다면 y는 그냥 숫자라고 생각해주면 z = 3x 와 같은 함수랑 크게 다르지 않다. 그러면 일차함수 미분꼴이니 결국 미분하면 y라는 상수값이 나온다고 생각해주면 편하다.

미분의 식은 다음과 같은데,

![2](https://user-images.githubusercontent.com/24144491/45867889-6b377580-bdbf-11e8-98cd-839c2792cfb5.JPG)

이 때, **```h```**란 아주 작은 수이다. 0.000000~~1정도라고 해야할까. 즉 **미분**은 **순간적인 변화율**인데, **f**를 **x**로 미분한 것은 x의 아주작은 변화가 생기면 f는 얼만큼 변화하는지에 대한 값. 다시말해, **```미분```**은 **입력 변수** 부근의 아주 작은(0에 매우 가까운) **변화**에 대한 **해당 함수 값의 변화량**이다.

다음과 같은 수식에서 미분값을 구하는 것 역시 어렵지 않다.

![3](https://user-images.githubusercontent.com/24144491/45867892-6bd00c00-bdbf-11e8-8de1-ec6d969b72e8.JPG)
![4](https://user-images.githubusercontent.com/24144491/45867895-6d013900-bdbf-11e8-9108-09fd540b203f.JPG)


## **3. 복합 표현식(Compound Expression), 연쇄 법칙(Chain rule), Backpropagation**
---
![5](https://user-images.githubusercontent.com/24144491/45867897-6d99cf80-bdbf-11e8-8a2c-17e1775cea7c.JPG)

자 이렇게 생긴 회로가 있다고 하자. forward pass 는 초록색으로 표시돼 있고, backward pass는 backpropation한 값들(바로 앞 뒤 관계만을 고려해 미분한 값)은 적색으로 표시돼있다. 이 회로를 찬찬히 살펴보자.

#### **Forward 부분**

![6](https://user-images.githubusercontent.com/24144491/45867899-6e326600-bdbf-11e8-97b6-f76adeceb9ad.png)

초록색들은 해당 값들 이다. 이 회로는 
> q = x + y
> f = q * z

의 구조를 가지고 있는데, 
> x = -2
> y = 5
> z = -4

가 입력된다고 하면, 
q 는 3이 되고 f는 -12가 된다.

#### **Backward 부분**
값들이 구성되기에 앞서서 각 회로의 한 부분(input, output)을 기준으로 해당 input으로 그 바로 다음 노드의 output을 미분한 것을 표시한 그래프는 다음과 같다.

![7](https://user-images.githubusercontent.com/24144491/45867900-6f639300-bdbf-11e8-9c5b-a7f2efc49504.png)

여기서 **`∂f / ∂x`** 는 어떻게 구할까?

x라는 변수가 바로 f로 가는게 아니라, x라는 변수가 q = x + y 라는 함수를 거치고, 그 함수가 f = q*z 로 이어진다.
즉, x 가 변하면 -> q 가 변하고 -> f 가 변하는 것이다.

따라서 **f를 미분**할 때도,

**x**가 변함에 따라 **q**가 변하는 것과 그렇게 바뀐 **q**가 변함에 따라 **f**가 변함을 같이 고려해 주어야 하므로 다음과 같은 식이 나온다. 이게 바로 **Chain rule****(연쇄법칙)**이다.

![8](https://user-images.githubusercontent.com/24144491/45867902-6ffc2980-bdbf-11e8-83b8-314c8188f217.JPG)

그래서 이렇게 계산된 회로의 값들이 위의 빨간색으로 표시된 값들이 되는 것이다.

![5](https://user-images.githubusercontent.com/24144491/45867897-6d99cf80-bdbf-11e8-8a2c-17e1775cea7c.JPG)

이 **Backpropagation**계산 과정을 **`코드`**로 고치면 다음과 같다.
```python
# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
# now backprop through q = x + y
dfdx = 1.0 * dfdq # dq/dx = 1. And the multiplication here is the chain rule!
dfdy = 1.0 * dfdq # dq/dy = 1
```

## **4. Backpropagation에 대한 직관적인 이해**
---

Backpropagation은 굉장히 지역적인(local) 프로세스를 가진다. 앞의 회로의 backpropagation에 대해서 살펴봤듯이 각 회로의 앞 뒤만 고려해 gradient 식을 구해놓고, 실제로 backward를 할때 필요한 변수값만 넣어주면 바로 계산되기 때문이다. 연쇄 법칙을 통해 게이트는 이 그라디언트 값들을 받아들여 필요한 모든 그라디언트 값을 곱해주면 끝.


## **5. 모듈성 : 시그모이드 예제**
---

위의 회로(circuit)에 대한 예는 Backpropagation과 나중에 배울 Neural Network 구조를 이해하는데도 좋은 예시이다. 어떤 종류의 함수도 미분이 가능하다면 게이트로서의 역할을 할 수 있다. 그의 한 예로 시그모이드함수를 살펴보자.

![9](https://user-images.githubusercontent.com/24144491/45867904-712d5680-bdbf-11e8-8576-a5eb3c318622.JPG)

- w = [w0,w1,w2]
- x = [x0,x1]

일 때, 다음과 같은 회로를 만들 수 있다.

![10](https://user-images.githubusercontent.com/24144491/45867906-71c5ed00-bdbf-11e8-9f5a-bff0cdf01e24.JPG)

- 이 회로 역시 초록색은 forward pass 한 값이고, 빨간색은 Backprop을 하면서 계산된 값이다. 

좀 더 이해를 돕기 위해서

![12](https://user-images.githubusercontent.com/24144491/45867913-74284700-bdbf-11e8-944a-ac81e6e5ca65.png)

- 검은색 부분은 x = w0 * x0 + w1 * x1 + w2
- 빨간색 부분은 σ(x) = 1 / (1 + exp(-x))

그런데 여기서 시그모이드 함수를 x에 대해서 미분을 하면

![13](https://user-images.githubusercontent.com/24144491/45867916-75597400-bdbf-11e8-89fa-a51eaa9de8cd.JPG) 

이 된다.

따라서 우리는 이 함수의 backprop 과정을 다음과 같이 간추릴 수 있다.

```python
w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2] # 위의 설명에서 x와 같음
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit
```
구현 팁(protip): forward pass 회로(?)나 함수들을 backprop을 쉽게할 수 있는 함수로 잘게 분해하는 것은 언제나 도움이 된다고 한다. 위의 코드에서 dot과 f함수를 만든 것처럼.


## **6. Backprop 실제 : 단계별 계산**
---

또 다른 예제를 통해서 Backprop을 확인해보자.

![14](https://user-images.githubusercontent.com/24144491/45867918-768aa100-bdbf-11e8-9acd-da9bcd313e95.JPG)

이런 함수가 있으면 다음과 같이 forward와 backprop을 계산하는 코드를 짤 수 있다.
```python
x = 3 # example values
y = -4

# forward pass


sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   		 #(1)
num = x + sigy # numerator (분자)									#(2)
#--------------------------------------------------------#
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator 		#(3)
xpy = x + y                                              		#(4)
xpysqr = xpy**2  												#(5)
den = sigx + xpysqr # denominator(분모)                  			#(6)
invden = 1.0 / den                                       		#(7)
#--------------------------------------------------------#
f = num * invden # done!                                 		#(8)
```
```python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```

여기서 몇 가지 주의할 점들은 다음과 같다.
- (1) forward 변수들 **cache(저장)**하라.
- (2) **Gradients**를 **더하라**.

(1) cache를 해둬야 backprop에 쓸 수 있으니 저장해야하고
(2) dx의 backprop를 보면, (4),(3),(2) 에서 각각 x에 대한 미분 값들이 나오는데 모두 x의 변화에 대한 f의 변화량이므로 **add** 해주어야 한다. 따라서 backprop 코드를 짤 때 역시 **`+=`**을 사용하도록 하자.


## **7. 역방향 흐름의 패턴**
---

![15](https://user-images.githubusercontent.com/24144491/45867919-768aa100-bdbf-11e8-97eb-c5d1c8eaddcc.JPG)

여기서 backward flow를 좀 더 직관적으로 봐보자.

- **(1)** **add gate** : add gate는 단순히 더하는 gate인데 f = x+y 라는 식이 있다면 결국 df / dx = 1, df / dy = 1이다. 단순히 더하는 일은 이전의 backprop의 결과를 그대로 가져오면 되고 변화되는 것은 없다.

- **(2) max gate** : add gate와 달리, max로 뽑힌 변수의 미분값은 1이고, 뽑히지 않은 변수의 미분값은 0이된다. 예를들어, q = max(z,w) 라는 식이 있으면, z = 2, w = -1 일 때, z>w이므로 q = z 가 되고, 이때 dq / dz = 1, dq / dw = 0이 된다. max 값이 아닌 변수로 미분을 한다면 그 미분값들은 다 0이 되고, 오직 max로 뽑힌 변수만 미분값이 1이므로 그 전에 backprop 계산한 값을 그대로 가져오면 된다.

- **(3) multiply gate** : k = x * y 라는 값이 있다면 미분 값들은 dk / dx = y, dk / dy = x 가 되므로 서로 switch가 된다. 

그런데 만약, **`multiply gate`**에서 x와 y에 대한 값들의 차이가 크다면 어떤 일이 벌어질까? x = 10000이고 y = 2일때, dk / dx = 2 이지만, dk /dy = 10000이 된다. 그런데 미분값으로 x와 y에 대한 조정을 하는 거니까 작은 변수는 엄청 크게 조정이 되게 되고 큰 변수는 오히려 엄청 작게 조정이 될 것이다. 

일반적인 예를 생각해본다면 수없이 많은 X의 features 들이 서로 다른 값들을 가지고 있다면, **F = WX** 에서 W를 업데이트하기 위해 X를 사용할텐데 엄청 큰 수들이 있으면 큰 수와 multiply된 weight들 역시 엄청 크게 조정이 될 거고 이를 보완하려고 learning_rate(=step size)를 엄청 조정해야할 것이다. 또 그렇게 하면 학습 속도가 느려지게 되는 문제도 생기게 될테고(그 밖에도 몇개의 문제가 더 생길 수 있다)... 

따라서 우리는 이런 문제들을 방지하기 위해 **feature**(X, input)들에 대한 **`preprocessing`** 과정을 거치게 되고 이 과정이 매우 중요한 역할을 하게 된다. (이 때는 scale 조정)



## **8. 벡터 기반의 그리라디언트 계산** 
---

이제 single variables 이 아닌 matrix(행렬), vector, tensor의 연산으로 확장해보자.

**Matrix-Matrix multiply gradient**

```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```

행렬의 연산에 있어서 가장 중요한 것은 행렬의 shape이다. 이해를 돕기 위해 다음의 코드 결과를 보자.

![16](https://user-images.githubusercontent.com/24144491/45867920-77233780-bdbf-11e8-8eaf-898d66753ee5.JPG)

D = W.dot(X) 인데 ``` dD / dW ```는 단순히 X가 와버리면 W - ( dD / dW ) 의 연산이 불가하다. 따라서 shape을 맞춰줘야하고, dD를 dW로 미분한다는 것 역시 dW 안에 있는 각각의 w0, w1, w2, ... ,w_(W.size-1) 에 대해 각각 다 미분해 준다는 얘기니까 dW = dD.dot(X.T)가 된다. 이 부분에 대해서 이해가 잘 안된다면 W = [3 x 2 ], X = [2 x 3] , D = W.dot(X)를 한 예시로 생각해서 각 dot의 결과(forward pass)가 무엇인지, 그것을 각각 w11, w12, w21, ... , w32 로 D 를 미분한 값들이 무엇인지 생각해보자. 

#### **추천 링크**
- [Vector, Matrix, and Tensor Derivatives](http://cs231n.stanford.edu/vecDerivs.pdf)
- [행렬의 미분 - 데이터 사이언스 스쿨](https://datascienceschool.net/view-notebook/8595892721714eb68be24727b5323778/)


## **요약**
---

- **Gradient**의 의미를 알았다.
- 회로를 통한 **flow, backwards pass, backprop**를 직관적으로 이해했다.
- **Backprop** 과정에 있어서 **staged computation**이 중요하다는 것을 알았다.
- 학습하기전 **feature** 들의 **preprocessing** 과정이 중요하다는 점을 발견했다.

## **느낀점**
---
 cs231n 강의관련 2번째 포스팅이 끝났다. 확실히 이해하고 적용하는데에는 큰 시간이 걸리지 않지만, 이것을 다른 사람에게 표현하고자 하니 생각보다 시간이 오래 걸린다. 또 쉽고 정확하게 전달하고자하는 욕심이 있어서 이것저것 더 찾고 생각하게 돼 더 많은 시간을 공부한다. 글로 주저리 주저리 쓰지말고 정리해보자.

- 추가로 포스팅 할 내용들 : (1)**feature preprocessing**, (2)**행렬 계산**, (3)**Backpropagation** API 구성 뜯어보기, 구현해보기
- markdown 글쓸 때 강조나, 이런 것들 단축키가 있다는 것 처음 앎.**(haroopad** 사용 중)
- 스크린샷 붙여넣을때 git issue 들어가서 해야하고 각 캡처한거 바로바로 못 넣어서 조금 아쉬움.
- 곧 **Paper**들도 봐야하는데 언제 시작하지라고 생각하지말고 포스팅 끝나고 당장 뽑아서 보기 시작.

사실 글을 쓸 때, 누군가가 본다는 생각으로 적으니까 어디서부터 어떤 부분까지 설명해야할 지 참 난감할 때가 많았다. 그래도 상세히 적는다고 적긴했는데 그러니 너무 진도가 안나가는 부분도 있고, 그래도 하나하나 설명하는 것이 바로바로 나왔다면 그것조차 빠른 시간 안에 할 수 있었을테니 결국 나의 실력 부족. 많은 것을 보고 많이 적고 물어보고 공부해보자.
