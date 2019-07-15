---
layout: post
title: "[CS231n] 강의노트 : 신경망 Part 2 (Neural Networks)"
subtitle: "cs231n 강의노트, data preprocessing, weight initialization, batch normalization, regularization, loss functions" 
categories: cs231n
tags: cs231n dl
img: stanford-univ.jpg

comments: true
---

## 목표

> **데이터 전처리, 가중치 초기화, 배치 정규화, 정규화, 손실함수의 이해**

## 공부기간

> **2018.09.25.화**

## 참고자료

- [CS231n 강의노트 - 신경망 part 2](http://cs231n.github.io/neural-networks-2/)
- [CS231n 강의노트 - 신경망 part 2 - 한글번역본 (AI Korea)](http://aikorea.org/cs231n/neural-networks-2-kr/)



# 본문
---

 이번 장에서는 신경망 모델에 넣을 데이터를 전처리하는 과정(Data Preprocessing), 신경망 모델에서 초기 가중치를 초기화하는 방법, 배치를 정규화하는 방법, 모델을 정규화하는 방법 그리고 손실함수를 어떻게 구성할지에 대해서 자세히 다룰 것이다.


## 목차

1. **데이터 전처리** (Data Preprocessing)
2. **가중치 초기화** (Weight Initialization)
3. **배치 정규화** (Batch Normalization)
4. **정규화** (Regularization)
5. **손실 함수** (Loss function)
6. **요약** (Summary)


## **1. 데이터 전처리 (Data Preprocessing)**
---

데이터 전처리에 앞서 기본적인 `X` (inputs = N 개의 features)의 형태는 [N x D] 의 사이즈로 구성된다. (N은 데이터의 수, D는 차원 = features의 수)

### Scale 조정 ( Normalization ㅡ )
> 이 부분은 CS231n의 강의노트보다 이해하기 더 쉽고 자세한 설명이 있는 링크를 가져왔다.

Scale 조정은 (1) Standard Scaler, (2) Robust Scaler, (3) Minmax Scaler, (4) Normalizer 4가지 방법이 있는데 [Scikit-Learn의 전처리 기능](https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/)을 통해 각 Scaler가 어떻게 구성되어있는지 참고하길 바란다.


### PCA & Whitening

**`PCA`**

데이터를 정규화 시키고 공분산(Covariance) 행렬을 만든다. 공분산 행렬이란 각 구조간 상관관계를 말해주는 행렬이다. 공분산을 계산하는 코드는 다음과 같다.

```python
# X = [N x D]의 input 일때,
X -= np.mean(X, axis = 0) 
cov = np.dot(X.T, X) / X.shape[0] 

```

- `cov`의 행렬의 (i, j)번째 원소는 두 차원(feature)간의 상관관계 정도다.


> **`예를들어,`** X 의 size가 [3 x 5], feature가 5개인 3개의 데이터 값을 이루어진 X라는 input이 있다고 하자. X.T 의 size는 [5 x 3] 이고, (X.T).dot(X) 는 [5 x 5]를 가지게 되는데, X.T는 생각해보면 feature가 행이되고, 열이 각 데이터 값이다. 그럼 둘의 dot product 결과물은, X.T의 i번째 행의 데이터(i번째 feature의 모든 데이터 값)과 X의 j번째 열의 데이터(j번째의 feature의 모든 데이터 값)의 곱이다. 

- 대각선 상의 값들 (diagonal of this matrix)는 각 feature 의 분산이다. (i, j) (단, i=j)

> 위의 **`예시`**에서 X.T의 i번째 행( i번째 feature의 모든 값 )과 X의 j번째 열의 데이터(j번째의 feature의 모든 데이터 값)의 dot product인데 i = j이므로, 같은 feature들의 데이터들이 곱해지면서 제곱의 의미가 된다. **`(데이터값 - 평균) / (데이터 수) = 분산`**.

- `cov` 행렬은 symmetic 이고, positive semi-definite 하다.
[]()

따라서 우리는 SVD factorizaiton을 다음과 같이 구할 수 있다.

``` python
U, S, V = np.linalg.svd(cov)
```

- U는 아이젠벡터(eigenvectors), 정규직교(orthonomoral) 벡터 (norm of 1, and orthogonal to each other) = basis vector 
- S는 1-D 1차원배열 (특이값의)

X를 고유기저(eigenbasis)로 사상(projection)시킴으로써 데이터 간의 상관관계를 없앨 수 있다.

```python
Xrot = np.dot(X,U)
```

또한 `np.linalg.svd`는 `U`행렬의 column vector는 각 벡터의 상응하는 아이젠밸류의 내림차순으로 정렬된다. 따라서 상위 중요한 몇 개의 vector들만 이용하여 차원을 축소하는데 사용할 수 있다.

```python 
Xrot_reduced = np.dot(X, U[:, :100]) # Xrot_reduced -> [N x 100] 
```

따라서 이 결과는 [ N x 100 ] 크기의 데이터로 압축되고, 데이터의 분산(variance)가 가능한 큰 값을 갖도록하는 100개의 차원이 선택된다. 이러한 기법을 **Principal Component Analysis(PCA)**,**차원 축소 기법**이라고 부른다.

**`Whitening.`** 기저벡터 (eigenbasis) 데이터를 아이젠벨류(eigenvalue)값으로 나누어 정규화 하는 기법이다. 이 변환의 **기하학적 해석**은 만약 입력 데이터의 분포가 **multivariable gaussian 분포** 라면 이 데이터의 평균은 0, 공분산은 단위행렬(I)인 정규분포를 가진다. 

```python
Xwhite = Xrot / np.sqrt(S + 1e-5)
```

> 1e-5와 같은 작은 수가 아닌 **더 큰 수를 분모에 더하는 방식**으로 스무딩(smoothing)효과를 추가하여 이런 **노이즈를 완화(?)**시킬 수 있다.

![1](https://user-images.githubusercontent.com/24144491/46000228-f166fb00-c0e2-11e8-9f88-b39e5980cc73.PNG)

이 기법들은 `CIFAR-10` 이미지에서도 적용할 수 있다. 학습 데이터세트 사이즈는 [50000 x 3072] 이고, [3072 x 3072]의 cov 행렬을 계산하고 SVD들 위와 같은 코드로 구할 수 있다.

![2](https://user-images.githubusercontent.com/24144491/46000229-f166fb00-c0e2-11e8-9012-ec4034965745.PNG)

왼쪽부터 1,2,3,4번 그림이라고 하자. **1번 :** 49개의 이미지 데이터이다. **2번 :** 3072개의 eignevetors 중 상위 144개의 아이젠 벡터이다. 3072개 차원의 벡터로 이미지를 다 표현하는 것보다 144개의 차원(eigenvectors, 새로운 features)로 이미지를 다시 표현할 수 있고, 이런 features들의 합으로 이미지를 표현하면 **3번**째 그림과 같다. **4번 :** whitening 시킨 결과인데, 각 차원의 분산은 동일한 길이를 가진다.

### In practice.
 Convolutional Networks에서는 사용하는 경우가 거의 없다고 한다.. 하지만 zero-center 하는 이 정규화 과정은 매우 중요한 기법이고, 이 부분은 많이 사용한다고 한다.


### **Common pitfall.**
- 전처리의 과정에서 얻은 통계적인 수치들은 학습데이터에서만 뽑고
- 그것을 test, validation dataset에 적용해야한다.
> 여기에 대한 이유는 조금 더 찾아봐야 할 것 같다.


## **2. 가중치 초기화 (Weight Initialization)**
---

 이번 절에서는 학습시 가중치를 적절히 초기화하는 방법에 대해서 알아보자

- **0으로 초기화하지 말것**
: 모든 Weight들을 0으로 초기화하면 역전파 과정에서 미분값이 동일하고, 동일한 값으로 업데이트 될 것. 따라서 뉴런들의 비대칭성(asymmetry)이 사라지게 되는 문제를 야기한다. 

- **small random number**
: 가능한 0에 가까운 값으로 초기화하자. 모든 가중치의 난수를 이용해 고유한 값으로 초기화 시킴으로써 파라미터들이 서로 다른 값으로 업데이트되고 결과적으로 다 다른 특성을 보이는 부분들로 분화될 수 있다. `W = 0.01 * np.random.randn(D,H)`로 구현할 수 있다. `randn`은 평균 0, 표준편차 1인 정규분포로 부터 얻은 값이다. 
 하지만 이 방법 역시 항상 좋은 성능을 보장하는 것은 아니다. 아주 작은 값으로 초기화 된 Weight의 경우 backprop 과정에서 미분값 또한 작은 값을 가지게 되고 느리게 혹은 local minima에 빠질 수 있는 문제가 있다.
 
- **분산 보정 `1/sqrt(n)`**
: `w = np.random.randn(n) / sqrt(n)`, 
var(w) = 2 / (n_inputlayer + n_outputlayer)로 초기화 할 것을 권장한다.
`w = np.random.rand(n) * sqrt(2.0/n)`을 이용하여 가중치를 초기화하라고 하며 ReLU 뉴런이 사용되는 신경망에서 최근에 권장되고 있는 방식이라고 한다.

- **희소 초기화 (Sparse initialization)**
: 보정되지 않은 분산을 위해서는 가중치 행렬을 0으로 초기화

- **bias 초기화 (initializing the bias)**
: 0으로 초기화하는 것이 일반적이다.

- **실전 'In parctice'**
: ` w = np.random.randn(n) + sqrt(2.0/n)` 으로 초기화하는 것이 요즘 추세라고 함.


## **3. 배치 정규화 (Batch Normalization)**
---

- [**Batch Normalization**](https://arxiv.org/abs/1502.03167)

이 논문에 이와 관련된 내용이 잘 나와있지만, 배치 정규화는 신경망의 활성화함수를 거친 값이 unit gaussian distribution이 되게 하는 기법이다. 일반적으로 CNN에서 fully - connected layers 다음이나 비선형레이어 전에 배치정규화를 넣어 구현한다고 한다. 신경망을 구현할 때 많이 쓰인다고 한다.
> 요즘은 fully connected layer을 쓰지 않는다는데, 관련된 자료를 더 찾아봐야겠다.



## **4. 정규화 (Regularization)**
---

**L2 정규화.** 가장 흔히 쓰이는 유형으로 모든 weight의 제곱값의 합을 이용한다. 그리고 적절한 정규화율인 **λ(lambda)**를 곱해주면 끝. 또 미분을 편하게 하려고 앞의 1/2를 곱해주어 `(1/2) λ w^2`의 정규화 term이 완성된다.
>The L2 regularization has the intuitive interpretation of heavily penalizing peaky weight vectors and preferring diffuse weight vectors. (L2 reguralization은 큰 값이 많이 존재하는 가중치에 제약을 주고, 가중치 값을 가능한 널리 퍼지도록 하는 효과를 주는 것으로 볼 수 있다)

backprop 해줄때는 `W += - lambda * W`와 같이 구현해주면 끝.

**L1 정규화.** 단순히 W에 절댓값을 씌운것이 L1 정규화 텀이다. L1의 정규화 텀과 L2의 정규화 텀을 합치면 다음과 같은데 `λ1∣w∣ + λ2w2` 이것을 Elastic net 정규화라고 부른다. L2 정규화와 달리 L1 정규화는 정확히 0으로 혹은 0에 가깝게 weight를 업데이트할 수 있다. 보통 L2 정규화를 더 많이 사용한다고 한다.(L1 보다 성능이 좋아서)

**Max norm constraints.** weight에 upper bound를 정해 weight가 너무 크게 설정되지 못하게 막아 미분값도 제한된다. 실전에서는  모든 뉴런의 W에 대해  ∥W∥_2 < c를 만족하게 한다. 이때 c는 3 또는 4로  설정한다. 어떤 사람들은 이 텀을 이용해 정규화의 성능향상을 보였다고 한다. 장점은 학습률(learning_rate)를 크게 잡더라도  네트워크는 "explode" 점점 커지지 않게 된다.(당연히 bound 했으니)

**Dropout.** 드롭아웃은 엄청 효과적이고, 간단하다. L1, L2, maxnorm과 상호보완적이다. 학습중에 P만큼의 확률로 그 뉴런을 활성화시키거나 0으로 만들어버린다.

![4](https://user-images.githubusercontent.com/24144491/46000233-f166fb00-c0e2-11e8-8f8d-69dc052fed2f.PNG)

3층 구조의 신경망에서 **inverted dropout**을 구성하면 다음과 같다.

```python
p = 0.5

def train_step(X):
	# X contains the data 
    
    # forward pass
    H1 = np.maximum(0,np.dot(W1,X) +b1)
    U1 = np.random.rand(*H1.shape) < p / p	#first dropout mask /p!
    H1 *= U1 #  drop
    H2 = np.maximum(0, np.dot(W2,H1) +b2)
    U2 = np.random.rand(*H2.shape) < p / p 	#Notice /p! <- inverted dropout
    H2 *= U2 # drop
    out = np.dot(W3, H2) + b3
   
	# backward pass  
    dout = np.random.randn(*out.shape)
    dW3 = dout.dot(H2.T)
    dH2 = W3.T.dot(dout)
    db3 = 1

    dH2 *= U2  	# dropout backprop
    
    dW2 = dH2.dot(H1.T)
    dH1 = W2.T.dot(dH2)
    db2 = 1
    
    dH1 *= U1	# dropout backprop
    
    dW1 = dH1.dot(X.T)
    dX = W1.T.dot(dH1)
    db1 = 1
    
    # perform parameter update 
    # loss function이 어떻게 정의될지 모르지만
    # loss fucntion을 미분한 값을 dLoss 라고하면
    W1 -= learning_rate * dLoss * dW1
    W2 -= learning_rate * dLoss * dW2
    W3 -= learning_rate * dLoss * dW3
    b1 -= learning_rate * dLoss * db1
    b2 -= learning_rate * dLoss * db2
    b3 -= learning_rate * dLoss * db3
 
def predict(X):
	H1 = np.maximum(0, np.dot(W1, X) + b1)
    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    out = np.dot(W3, H2) + b3
```
**`train_step`** 함수를 보면, 2번의 dropout을 실행한다(1,2번째 히든레이어에서). 
**`backward pass`** backpropagation 과정은 생략되어 있지만, 앞의 절을 참고하면 충분히 구현할 수 있다. dropout back prop하는 과정을 잘 생각해보면, 결국 활성화 된 뉴런 부분의 가중치(weight) 및 bias를 업데이트 해주면 되니까, 각 dropout된 결과를 저장한 U1, U2를 그냥 곱해주면 된다. (미분할 때 활성화 되지 않는 부분은 0이 되니까 결국 그 부분의 weight는 업데이트 안해줘도 됨) 그리고 일련의 과정들을 back prop 과정들을 그냥 짜보았다.(돌려보진 않았지만 맞을 것 같다, shape을 잘 맞춰 넣는다면 말이다)

#### Dropout 관련 참고 자료
- [Dropout forward-backward pass 과정 설명 및 코드](https://wiseodd.github.io/techblog/2016/06/25/dropout/)
- [Dropout 논문 by Srivastava et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- [Dropout Training as Adaptive Regularization](http://papers.nips.cc/paper/4882-dropout-training-as-adaptive-regularization.pdf): “we show that the dropout regularizer is first-order equivalent to an L2 regularizer applied after scaling the features by an estimate of the inverse diagonal Fisher information matrix”.



**Theme of noise of forward pass.** Dropout은 신경망에 확률(stochastic)의 개념을 가져왔다. testing 과정에서 노이즈가 감소하게 되는데, 확률 p만큼 노이즈가 제거될 가능성이 있어서이기도 하고 또는 각 dropout을 적용해서 각 결과들을 평균하면서(ensemble)이기도 하다고 한다. [이 방향과 관련된 다른 연구 자료 : Dropconnect](https://cs.nyu.edu/~wanli/dropc/). Convolutinoal Neural Network(CNN)은 이 부분에 있어서 여러가지 풀링기법과 데이터 변환(augmentation)기법으로 노이즈 문제를 해결하기도 한다. 이부분은 CNN을 배우면서 추후에 다룰예정이다.


**Bias 정규화.** 실전에서 역시 bias에 정규화를 적용할 때 성능저하를 보인 케이스는 극히 드물다고 한다. weight와 비교해서 bias는 상당히 규모가 작기 때문에 loss를 줄일 수 있을 때 참고해보는 정도라고 한다.

**레이어마다 정규화.** 마지막 출력 레이어를 제외하고 레이어를 각각 따로 정규화 하는 것은 일반적인 방법이 아니다. 레이어 별 정규화를 적용한 논문수도 상대적으로 매우 적은 편이다.

**실전.** 하나의 전역적인 L2 정규화 텀을 사용하고 모든 레이어 다음에 dropout을 결합하는 경우가 일반적이다. dropout의 확률 p의 default는 0.5 이지만 검증때 조금씩 바꿔볼 수도 있다.


## **5. 손실 함수 (Loss function)**
---

일반적인 손실함수를 구성하는 요소는 크게 2가지다.
- **(1) Data Loss** 
: 실제 값(우리가 예측해야되는 값=truth value)과 예측한 값(모델로부터 나온 결과값=predicted value)간의 차이 (error, loss)
- **(2) Complexity of Model Loss**
: 모델의 복잡도에 대한 패널티. 정규화를 위한 term.

**`Classification(분류).`** 

![5](https://user-images.githubusercontent.com/24144491/46000234-f1ff9180-c0e2-11e8-8152-977c1688c59a.PNG)

왼쪽은 SVM의 Loss function. 오른쪽은 크로스 엔트로피를 이용한 Softmax classifier의 loss function 이다.

**Problem:클래스의 수가 많을때.** 만약 구분해야하는 클래스 수가 많을경우(예를 들어, 영어 사전, ImageNet의 22000가지의 종류들), **Hierarchical Softmax**를 이용하면 도움이 될 수 있다. 이것은 라벨을 트리로 분해시켜 각 라벨들은 하나의 path가 된다. 그리고 트리의 모든 노드에서 구분을 애매하게 만드는 branch들을 train 시킨다. 이런 트리 구조는 효과적이라고 한다 (어디까지나 역시 problem(or data) - dependent)

**분류의 속성.** SVM Loss나 Softmax C.의 Loss 에서 `y_i` 는 정답 라벨이다. 그런데 만약 이런 맞춰야 될 라벨이 여러개의 클래스를 가지는 라벨이라면? 인스타(instagram)의 사진을 한 번 생각해보자. 사진 하나에 엄청나게 많은 해쉬태그들이 있다. 이 경우에는 각 클래스당 binary classfier를 독립적으로 만드는 방법이 있다.  

![6](https://user-images.githubusercontent.com/24144491/46000235-f1ff9180-c0e2-11e8-9b47-6427c71c47f4.PNG) 

- j는 j번째 클래스
- i는 i번재 example
- y_ij는 i번째 example의 j번째 클래스의 정답 라벨 (1 or -1)
- f_j는 score function으로 양의 값이면 j번째 클래스라는 거고, 음의 값이면 j번째 클래스가 아니라는 결과내는 함수

따라서 L_i (i의 example의 loss)는 j클래스가 맞는 example의 f_j값이 +1보다 작거나, j클래스가 아닌 example에서 f_j의 값이 -1보다 크면 loss가 생긴다.

이것을 Regression term을 이용해 나타낼 수도 있다.

![7](https://user-images.githubusercontent.com/24144491/46000236-f1ff9180-c0e2-11e8-8d78-b8ca0985470f.PNG)

클래스가 1또는 0일 확률을 위와같다고 하자. 그 확률이 0.5보다 높을때 각 클래스라고 하면, 이 확률의 우도의 로그는(log likelihood of this prob.) 다음과 같다.

![8](https://user-images.githubusercontent.com/24144491/46000238-f1ff9180-c0e2-11e8-8d33-f1b33025211f.PNG)

`∂Li / ∂fj` 는 `yij−σ(fj)`이다. 여기서 학습을 위해서는 fj 역시 그에 해당되는 W와 b로 미분하고 곱해서 업데이트 해주면 된다.  


**`Regression(회귀).`**

회귀분석의 Loss도 마찬가지로 data와 실제로 예측해야할 값의 차이를 2가지로 나타낼 수 있다.

![9](https://user-images.githubusercontent.com/24144491/46000227-f0ce6480-c0e2-11e8-9b7b-dfe08c1e273c.png) 

왼쪽은 **Error(= 실제값과 예측값의 차이)**의 제곱이고, 오른쪽은 Error의 절댓값이다. 학습할 때 미분하는 것은 그렇게 어렵지 않으므로 쉽게 Gradient를 얻을 수 있을 것이다. (L2가 상대적으로 더 안정적이라고 한다)



## **6. 요약 (Summary)**
---

- 데이터 전처리에서 scale 조정은 정규화하고 규모를 [-1, 1] 사이 값으로.
- 가중치 초기화는 `w = np.random.randn(n) * sqrt(2.0/n).`
- 정규화는 L2 regularization + dropout (the inverted version)
- batch normalization 사용
- the most common loss function들을 살펴봄


# 마무리
---

- 블로그에 댓글기능 추가해야 함
- Xavier 관련해서 찾아보기
- Batch Norm에 대해 더 찾아보기 - Fully Connect Layer 관련해서도
- 강의노트 정리 끝나면 강의도 보고 정리해서 글올리기
