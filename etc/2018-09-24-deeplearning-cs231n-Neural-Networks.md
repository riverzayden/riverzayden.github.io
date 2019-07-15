---
layout: post
title: "[CS231n] 강의노트 : 신경망 Part 1 (Neural Networks)"
subtitle: "neuron, activation function, neural net architecture, neural networks"
categories: cs231n
tags: cs231n dl
img: stanford-univ.jpg
comments: true
---

## **목표**
> **' 신경망 (Neural Network) '** 의 구조 이해

## **공부기간**
> 2018.09.24.월 

## **참고자료**
- [CS231n 강의노트 Neural Networks part 1](http://cs231n.github.io/neural-networks-1/)
- [CS231n 강의노트 한글 번역(AI-Korea)](http://aikorea.org/cs231n/neural-networks-1/)



# 본문
---
 오늘은 드디어 **hot**한 **신경망(Neural Networks)**에 대해서 알아볼 것이다. 글을 해석하는 과정에서 오류가 있을 수 있거나 주관적인 견해가 들어갈 수 있다. 이를 참고하면서 읽어주길 바란다.
 
## 목차
1. **소개**
2. **뉴런(neuron) 모델링**
3. **신경망 구조 (Neural Network Architectures)**
4. **요약**


## **1. 소개**
---

### **`CIFAR-10`**의 경우
 이 **score function**( = **`s = Wx`** )를 사용해 각 카테고리(종류)마다 다른 값(Score)을 계산했다. 
 
- **x** = [3072 x 1] 인 컬럼 벡터 (이미지의 모든 픽셀 값들) 
- **W** = [10 x 3072] 인 행렬

따라서  
 - **output(=s)**은 [10 x 3072] X [3072 x 1] = [10 x 1] 인 10개 클래스의 각 score value를 가지는 vector

### **`신경망(Neural Networks)`**의 경우
 이 **score function**( = **`s = W2 max(0,W1x)`** )를 사용해 각 카테고리(종류)마다 다른 값(s = score)을 계산한다.
 
 - **x** = [3072 x 1] 인 컬럼 벡터 (이미지의 모든 픽셀 값들) 
 - **W1** = [100 x 3072]
 - **W1x** = [100 x 3072] x [3072 x 1] = [100 x 1] 
 - **max(0, W1x)** = [100 x 1]의 100개의 vector 값들을 각각 0이랑 비교해 max 값을 취한 [100 x 1] vector
 - **W2** = [10 x 100] 의 행렬

따라서

 - **output(=s)** = [10 x 1] 인 vector 가 된다.

여기서 max 라는 함수가 없었다면 위의 계산은 W2 W1x 가 될텐데, 그렇게 되면 W2W1 = W3 라는 하나의 행렬로 나타낼 수 있고 이런 계산은 W3x 로 선형(linear)적이게 된다. 하지만 **max**함수가 있으므로 인해 score 함수는 nonlinear 하게 된다. 따라서 이 모델을 기준을 W1, W2를 학습시키려면, **`s = W2 max(0,W1x)`**함수를 W1, W2로 미분하는 **Gradient Descent**과정에서 연쇄법칙(chain rule)을 이용해야 할 것이다. 

만약, 3개의 층(layer)이 있다고 하자. ****`s = W3 max(0, W2 max(0,W1x))`**** 라는 score 함수가 있다면 W3, W2, W1를 학습할때, W3, W2, W1을 기준으로 각각 GD 과정을 거쳐야 한다. 이때 중간 층에 있는 neuron, hidden vector은 (= **W2 max (0, W1x)**) hyperparameter가 될 것이다. (이 hyperparameter 에 대해서는 나중에 살펴보기로 하자)

글과 식으로 신경망(Neural Networks)를 설명하는데는 한계가 있으니 시각화된 모델과 추가된 설명으로 천천히 구조를 살펴보기로 하자.


## **2. 뉴런(neuron) 모델링**
---

 - **생물학에서 얻은 동기와 연결고리** (Biological Motivation and connections)
 - **선형 분류기와 같은 하나의 뉴런** (Single Neuron as a linear classifier)
 - **활성화 함수** (activation function)

### - **생물학에서 얻은 동기와 연결고리**

뇌의 기본적인 계산 유닛은 **뉴런(Neuron)**이다. 약 86십억개의 뉴런이 있으며 10의 14제곱 ~ 10의 15제곱 개의 **시냅스(Synapses)**들이 있다. 아래의 다이어그램을 보자.

![1](https://user-images.githubusercontent.com/24144491/45941270-530e6300-c018-11e8-8c3e-2d2daf04fb4a.PNG)

- 왼쪽은 뉴런의 그림이고 오른쪽은 이를 표현한 대표적인 수학적 모델이다.

먼저 뉴런 그림부터 보자. 뉴런은 그것의 **dendrites**로 부터 input을 받아들이고, 그것의 **axon**을 통해 output을 내보낸다. 그 axon은 **synapse**를 통해 다른 뉴런의 dendrities와 연결된다. 

 수학적 모델을 보자. 뉴런의 계산 모델

- **x0** = signal input = 어떤 axon으로부터 나온 output 값
- **w0** = synaptic strength 

w0(weight)는 학습이 가능하고 그것의 영향력을 조절(control)할 수 있다고 하자. (excitory(양의 weight)나 inhibitory(음의 weight)로 학습 및 조절을 할 수 있을 것) 기본 모델에서, 그 dendrites는 signal(input)을 가져오고 cell body에서는 그런 synapse를 적절히 통과해온 signal들을 다 합하게 된다. 그리고 합해진 값이 axon을 통과하면서 또다른 값으로 바뀔 수 있다. (axon 역시 그 안에서 어떤 작용을 할 수 있으므로.) 다음과 같이 생각하면 이해하기 편하다.

```
input(x0) 
-> synapse(w0 = weight) 
-> dendrite(w0,x0) 				# dendrite는 두 값을 곱하는 함수
-> cell body(sum(wixi)+b) 			# 그렇게 곱해진 모든 값과 bias를 합함
-> axon(activation function(cell body output 값))	# axon역시 어떤 함수로 cell body의 결과값을 함수에 넣어 처리
-> output
```

다시 다음과 같이 정리하면
- **x0** = siganl
- **s0** = synapse strength
- **dendrite** = 입력된 두 값을 곱하는 함수
- **cell body** = 모든 입력 값들을 더하는 함수
- **axon** = 활성화 함수 (정의하기 나름)

이런 과정들을 다음과 같은 코드로 짤 수 있다.

```python
x = [1, 2, ... , 100] 		# 100개의 inputs
w = [2, 3, ... , 101]		# 100개의 synaptic strengths
bias = 0.02				# 적절한 bias
for i in range(100) : 		
 d = dendrite(x[i],w[i])
 cellbody += d

cellbody += bias
output = axon(cellbody)
```


**activation function**을 **sigmoid fucntion**이라고하고 이 뉴런의 모델을 좀더 다듬은 코드는 다음과 같다. (이때 sigmoid function은 **`σ(x)=1/(1+e^(−x))`** .)

```python
class Neuron(object):
  # ... 
  def forward(self, inputs):
    """ assume inputs and weights are 1-D numpy arrays and bias is a number """
    cell_body_sum = np.sum(inputs * self.weights) + self.bias
    firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid activation function
    return firing_rate
```

이런 뉴런들이 여러개 연결돼 있다는 것을 생각해보자. 이것이 바로 **신경망(Neural Network)**이다!

### - **선형 분류기와 같은 하나의 뉴런**

 위에서 살펴본 뉴런의 수학적 모델을 어디선가 본 듯하다. 이전 선형 분류기(linear classifier)에서 본 것처럼, 뉴런에는 input에서 활성화된(like, activate) 영역 또는 비활성화된(dislike, activation near zero)영역들을 가지고 있다. 뉴런의 Output에 대한 적절한 Loss function을 만들 수 있으므로 이 single neuron을 선형 분류기(linear classifier)와 같은 역할을 하는 뉴런으로 만들 수 있다.

#### **Binary Softmax classifier**
 **`예를 들어`**, 
 **σ(∑iwixi+b)** 의 값을 어떤 클래스의 확률 (**P(yi=1∣xi;w)**)이라고 생각해보자. Binary 문제이므로 class는 2개, 1또는 0이다. 그렇다면 다른 클래스의 확률은 **P(yi=0∣xi;w)=1−P(yi=1∣xi;w)**가 될것이다. 따라서, Cross Entropy Loss를 정의하여 Optimizatino을 하면된다. (이때 0 또는 1을 구분하는 지점은 확률값이 0.5 이상일 때라고 정할 수 있다)

#### **Binary SVM Classifier**
이전에 살펴본 것처럼 support vector machine 으로 loss를 정의할 수도 있다.

#### **Regularization interpretation**
위의 softmax , SVM classifier에서 모두 정규화 텀을 추가할 수 있다.


> 다시 말해, **Single Neuron 은 하나의 Binary Classifier으로 구현될 수 있다.**

### - **활성화 함수 (activation function)**

앞의 뉴런의 수학적 모델에서 활성화 함수는 cell body의 값을 input으로 가지는 함수이다. 활성화 함수는 말 그대로 **함수**이다. 따라서 어떤 함수든 다 activation function이 될 수 있다. 더 많은 함수를 알고 싶다면 [다음링크](https://en.wikipedia.org/wiki/Activation_function)를 참고하라.

이 강의에서 소개된 함수들은 다음과 같은데,

- Sigmoid
- Tanh
- ReLU
- leaky ReLU
- Maxout

> **TLDR**: “What neuron type should I use?” Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of “dead” units in a network. If this concerns you, give Leaky ReLU or Maxout a try. Never use sigmoid. Try tanh, but expect it to work worse than ReLU/Maxout.

sigmoid는 weight vanish 등의 문제가 발생하니 사용안하고 Leaky ReLU/Maxout 위주로 사용되니 이 2가지만 살펴보겠다.
- **leaky ReLU**
- **Maxout** 

### **Leaky ReLU**

ReLU의 변형으로 

```
f(x) =  ax  ,(x<0)
	x   ,(x>=0) # where a is a small constant.
```

![2](https://user-images.githubusercontent.com/24144491/45941271-53a6f980-c018-11e8-9040-229e8cac3641.PNG)


왼쪽 그래프가 ReLU이고 오른쪽 그래프가 LeakyReLU이다. 이 activation function을 이용하면 항상 그런 것은 아니나, 대부분 성능이 좋다. 음의 값일 때도 미분값이 a로 살아있고, 미분할 때 역시 1 또는 a이니 엄청 빠르게 학습된다는 장점이 있다.

### **Maxout**

**Maxout_f = ```max((w1_T x + b1 , (w2)_T x+b2)```**
- T는 transpose
- ReLU에서는 w1 = 0 행렬, b1 =0인 경우임.
- 2개씩의 parameter(weight , bias)들이 더 생기는 문제가 있음.

> (사실 maxout을 쓴 경우를 별로 본적이 없어서 자주 쓰이는지는 모르겠다.)


## **3. 신경망 (Neural Networks) 구조**
---

### **Layer-wise organization**

![3](https://user-images.githubusercontent.com/24144491/45941272-53a6f980-c018-11e8-8706-5608de106a84.JPG)

신경망(Neural Networks)은 위의 그림처럼 여러가자의 뉴런이 각 층(layer)을 이루면서 서로 연결 되어 있는 구조를 가진다. 완전히 다 연결된 층(fully-connected layer)은 가장 흔한 신경망의 층 구조인데, 이는 이전 층의 한 뉴런은 다음층의 모든 뉴런과 연결되어 있는 구조를 말한다. 위의 경우가 완전히 다 연결된 층의 예시이다.

몇 가지의 이슈들을 짚고 넘어가자.

**Naming conventions.** N-layer (n개의 층)을 가진 신경망을 생각해보자. 1층짜리 구조는 input 과 output이 직접 연결된 구조로, 중간의 숨겨진 층(은닉층 = hidden layer)이 없는 구조다. 반면 위 그림의 예시처럼 왼쪽은 2층짜리 구조, 오른쪽은 3개의 layer를 가지는 신경망 구조이다. 이것을 생각하면 SVM은 흔히 1층짜리 신경망이라 볼 수 있다. 흔히 신경망을 "인공신경망"(Artificial Neural Network, ANN)나 "Multi-Layer Perceptrons"(MLP)이라고도 한다. 또 많은 사람들이 뉴런을 단위(unit)이라고 부르는 것을 좋아한다고 한다.

**Output Layer.** 마지막 output 층은 말 그대로 아웃풋, 결과값을 가진 층이다. 주로 class의 예측값이나, 관련 클래스일 확률 등의 real-value를 가진다.

**Sizing neural networks.** 흔히 신경망의 size를 측정하는 척도로 **뉴런의 수**나 **parameter의 수(Weight size)**이다. (parameter는 뉴런과 뉴런의 연결된 부분에서 (weight or  bias)가 있는 선이라고 생각하면 편하다)

> 위의 그림에서, 
> `왼쪽`구조는 4+2 = 6개의 뉴런이 있고, [3x4] + [4x2] = 20 개의 weights와 4+2 = 6개의 biases. 총 26개의 parameters가 있다. `오른쪽`구조는 4+4+1 = 9개의 뉴런과, [3x4] + [4x4]+[4x1] = 32개의 weights와 4+4+1의 biases. 총 41개의 parameters가 있다.

> 딥러닝에서는 대략 10-20개의 층이 있는 신경망을 볼 수 있는데 거기서 나오는 parameters의 수는 어마어마하게 많을 것이다. 후에 **`parameter sharing`**(각 층과 층 사이 weight들 공유)을 통해 효율적으로 연결할 수 있는 구조를 살펴볼 것이다.

### **Example feed-forward computation**

![3](https://user-images.githubusercontent.com/24144491/45941272-53a6f980-c018-11e8-8706-5608de106a84.JPG)

다시 이 구조를 살펴보자. 각 뉴런은 어떤 numeric(수치) 값이다. 이때 input vector는 [3x1]이 되어야한다. 또 각 층에 대한 weights 역시 하나의 Weight vector로 표현할 수 있다.

**`에를 들어`**
- **W1** = 1층 [4 x 3] 크기를 가지는 weight
- **b1** = 1층 bias vector
- **W2** = 2층 [4 x 4] 크기를 가지는 weight
- **b2** = 2층 bias vector
- **W3** = 3층 [1 x 4] 크기를 가지는 weight
라고 하면, **오른쪽 그림**의 3층짜리 신경망을 다음과 같이 표현할 수 있다.

```python
# import numpy as np
sigmoid = labmda x : 1.0/(1.0 + np.exp(-x))
h0 = np.random.rand(3,1)			# x = h0 = input
h1 = sigmoid(np.dot(W1,x) + b1)
h2 = sigmoid(np.dot(W2,h1) +b2)
h3 = np.dot(W3,h2) +b3			# h3 = output
```

> 위를 더 자세히 표현하면 이런 구조이다.

![4](https://user-images.githubusercontent.com/24144491/45942453-63750c80-c01d-11e8-9762-6f52e27205dc.PNG)

### **Representational power**
> Neural Networks work well in practice because they compactly express nice, smooth functions that fit well with the statistical properties of data we encounter in practice, and are also easy to learn using our optimization algorithms (e.g. gradient descent). Similarly, the fact that deeper networks (with multiple hidden layers) can work better than a single-hidden-layer networks is an empirical observation, despite the fact that their representational power is equal.

실전에서 신경망을 이용한 모델은 우리가 접하는 데이터의 통계적 특성과 잘 맞고, 부드럽고 간결하게 표현할 수 있고, 최적화(Optimization) 알고리즘을 사용하여 학습이 쉽기때문에 비교적 잘 작동한다. 또 깊은 네트워크일수록 더 표현할 수 있는 부분이 많아지기 때문에 단일 은닉층으로 구성된 신경망보다 더 좋은 결과를 낼 수 있다고 한다. (그렇다고 너무 깊게 쌓는 것도 좋진 않음. **data에 따라 역시 달라질 수 있다**)


다음의 참고자료를 보면 그 느낌을 더욱 선명하게 받을 수 있을 것이다.
- [Deep Learning book in press by Bengio, Goodfellow, Courville, in particular Chapter 6.4.](http://www.deeplearningbook.org/)
- [Do Deep Nets Really Need to be Deep?](https://arxiv.org/abs/1312.6184)
- [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)

### **Setting number of layers and their sizes**

**`Q.`** 그렇다면 **몇 층을 쌓고 각 층의 뉴런의 수는 얼마나 설정하는게 좋을까?** 
-  **층을 많이 쌓고, 뉴런을 늘리면 늘릴수록 (네트워크의 size를 크게 하면 할수록) 네트워크의 capacity는 증가한다. **

다음의 diagram를 보자

![5](https://user-images.githubusercontent.com/24144491/45941274-543f9000-c018-11e8-8207-05b2d5928c36.JPG)

복잡한 구조의 네트워크일수록 복잡한 표현을 할 수 있고, 실제로 학습시 더 낮은 loss를 만들 수 있는 weight들을 찾을 수 있다. 

- **하지만 동시에, 과적합될 수 있다.(Overfitting) **

**`예를들어`** 20개의 hidden neuron들이 **noise data**까지 잘 걸러내기 위해 학습을 진행했다고 하자. 그렇다면 그 noise data로 인해 위의 다이어그램처럼 빨간색 영역은 한 클래스로 다른 영역은 다른 클래스로 인식한다. 하지만 그 노이즈 데이터가 다른 데이터들과는 동떨어진 **outlier** 이다. (그러니 noise) 그러면 과적합된 모델으로는 정상적인 데이터를 잘못 분류하게 될것이다. 실제로, 3개의 hidden neuron으로 학습한 데이터가 test set에서 더 좋은 결과를 보여주었다고 한다.(generalization on the test set)

그렇다면 더 적은 뉴런을 사용해야할까?

- **과적합문제를 해결할 수 있는 많은 방법들이 있다. (L2, L1 Regularization, dropout, input noise etc.)**

**실전에서는 hidden neuron을 줄이는 것보다 위의 방법들을 이용하는 것이 더 좋은 성능을 가져온다고 한다.** 그 이유로, 작은 네트워크 일수록 표현할 수 있는 부분이 적어지기 때문에 Loss 값 역시 비교적 적은 local minima(지역 최솟값 = 극솟값)를 가진다. 반대로 size가 큰 신경망의 경우 더 많은 local minima를 가지게 되는데, 이런 값들 중 실제(actual) 손실 측면에서보면 더 우수한 경우가 많다.(더 비슷한 값들) 실제로 학습된 결과들의 분산도 후자의 경우가 훨씬 더 작다고 한다. 

![6](https://user-images.githubusercontent.com/24144491/45941269-530e6300-c018-11e8-9074-c5e55792ebef.JPG)

- 여기서의 람다는 정규화의 hyperparameter로 람다 값이 높으면 높을수록 더 간단한 모델로 바뀌게 된다.


## **4. 요약**
---

- 뉴런(Neuron)의 구조와 그것의 수학적 모델을 알아봄
- 신경망(Neural Network)의 구조를 알아봄
- 활성화함수(Activation function)을 알아봄


# 마무리
---

- **신경망(Neural Network)**에 관한 시리즈는 총 3파트이다. (매일 한 파트씩 요약하는 것이 목표, 추석이라 새벽이나 자투리시간을 최대한 활용하여)
- 신경망, CNN까지 다 정리하면 딥러닝의 일련의 과정들을 **한 그림** 안에 표현해보도록 하자.
