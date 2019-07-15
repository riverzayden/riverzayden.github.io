---
layout: post
title: "[NLP] 밑바닥부터 시작하는 딥러닝2 - Ch4 : word2vec 개선" 
category: nlp
tags: dl nlp word2vec 자연어처리 Embedding 밑바닥부터 시작하는 딥러닝2
comments: true
img: nlp3.jpg
---





# 4. word2vec 속도 개선 

---

`밑바닥부터 시작하는 딥러닝2`의 **4장에서는** 지지난주에 봤던 [ch3. word2vec](<https://taeu.github.io/nlp/ch3_word2vec/>) 의 속도를 개선하기 위해 크게 2가지를 살펴볼 예정이다. 우선 코드를 돌리기에 앞서 GPU 연산을 위해 아래와 같은 코드를 실행한다. 책에 나와있는 코드에서 `np.add.at = np.scatter_add`부분이 에러가 나서 `class Embedding: ` 에 np.add.at 대신 np.scatter_add를 적어주었으니 혹시 CPU로 돌려야한다면 아래의 코드에서 Embedding 클래스의 np.scatter_add를 np.add.at으로 바꾸고, GPU를 false로 바꾼뒤 실행하기 바란다. 그리고 아래 코드를 보면 `cupy`를 이용해 GPU 연산을 수행하므로 cupy를 설치해야주어야 한다. 그 전에 자신의 설치환경에 맞는 cuda와 그에 맞는 cupy를 설치하는 것이 핵심! 자세한 것은 [cupy document](<https://docs-cupy.chainer.org/en/stable/>)를 보면서 설치했으니, 참고하시길!




```python
GPU = True
if GPU: # GPU
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    #np.add.at = np.scatter_add

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
else :
    import numpy as np
```

    [92m------------------------------------------------------------[0m
                           [92mGPU Mode (cupy)[0m
    [92m------------------------------------------------------------[0m



## 목차

___



4.1 word2vec 개선 1
- 4.1.1 Embedding 계층
- 4.1.2 Embedding 계층 구현

4.2 word2vec 개선 2
- 4.2.1 은닉층 이후 계산의 문제점
- 4.2.2 다중 분류에서 이진 분류로
- 4.2.3 시그모이드 함수와 교차 엔트로피 오차
- 4.2.4 다중 분류에서 이진 분류로 구현
- 4.2.5 네거티브 샘플링
- 4.2.6 네거티브 샘플링의 샘플링 기법
- 4.2.7 네거티브 샘플링 구현

4.3 개선판 word2vec 학습
- 4.3.1 CBOW 모델 구현
- 4.3.2 CBOW 모델 학습 코드
- 4.3.3 CBOW 모델 평가

4.4 word2vec 남은 주제
- 4.4.1 word2vec을 사용한 애플리케이션의 예
- 4.4.2 단어 벡터 평가 방법

4.5 정리



# 4.1 word2vec 개선 1

___



![p4-2](https://user-images.githubusercontent.com/24144491/59848681-b60f0980-93a0-11e9-987b-6fc8d97f5554.png)

지난 시간 살펴본 word2vec은 여전히 큰 2가지 문제를 안고있다. 예를들어 어휘수가 100만개라고 한다면 다음의 과정에서 상당한 메모리와 계산량이 필요하다.
- [1] 입력층의 원핫 표현과 가중치 행렬 W_in 의 곱 계산 : 4.1 임베딩 계층 구현으로 해결
- [2] 은닉층과 가중치 행렬 W_out의 곱 및 Softmax 계층의 계산 : 4.2 네거티브 샘플링으로 해결

4.1에서는 먼저 **Embedding** 계층 구현부터 살펴본다.



### 4.1.1 Embedding 계층
![p4-1-1](https://user-images.githubusercontent.com/24144491/59848679-b60f0980-93a0-11e9-9d19-238ce3a8b238.png)

> corrections : hidden layer shape = (1 x 100)

: 실제 단어의 원핫 표현과 W_in의 가중치 매트릭스간의 곱의 결과는 W_in[원핫 표현에서 해당 단어의 index 번째 행] 과 동일하다는 것을 알 수 있다.



### 4.1.2 Embedding 계층 구현
- `class Embedding`
  : **forward** 때는 index만 넘겨 은닉층(h)를 얻고
  ![p4-1-2](https://user-images.githubusercontent.com/24144491/59848680-b60f0980-93a0-11e9-90f3-42881ffd9c78.png)

  : **backward** 때는 dW 업데이트 해야할 index가 겹치는 경우가 있을 수 있으므로 '할당'이 아닌 '더하기'를 해야 한다.

    

```python
# 4.1.2 Embedding 계층을 구현을 이해하기 위한 기본적인 행렬 연산들
#import numpy as np
W = np.arange(21).reshape(7,3)
W
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11],
           [12, 13, 14],
           [15, 16, 17],
           [18, 19, 20]])




```python
print("W[2] : ", W[2])
print("W[5] : ",W[5])
idx = np.array([1,0,3,0])
print("W[[1,0,3,0]] = W[idx] : \n",W[idx])
```

    W[2] :  [6 7 8]
    W[5] :  [15 16 17]
    W[[1,0,3,0]] = W[idx] : 
     [[ 3  4  5]
     [ 0  1  2]
     [ 9 10 11]
     [ 0  1  2]]



```python
# 4.1.2 Embedding 계층 구현
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx] 
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.scatter_add(dW, self.idx, dout) # cpu는 numpy 일때, np.add.at(dW, self.idx, dout)
        return None
  
```

>  **[Warning]** 가중치 W와 크기가 같은 행렬 dW를 만들고 W에 더해주는 식으로 적용하려고 backward를 dW[...]=0으로 구현함 (Optimizer) 클래스와 조합해 사용하고자. 하지만 비효율적.바로 W에 dout을 빼주면 되니까. # Optimizer 부분 봐서 수정해볼 것.



# 4.2 word2vec 개선 2

___



은닉층까지의 계산은 index를 활용해 필요한 계산만 실시하므로 효율적으로 바꿨다고 할 수 있으나, 여전히 W_out 부분의 계산과 수 많은 어휘와 Loss를 구해야 하는 상황. 이를 해결하기 위해 `네거티브 샘플링`을 이용하고자 한다.



### 4.2.1 은닉층 이후 계산의 문제점
- 은닉층과 W_out (은닉층의 뉴런수 x 어휘수)간의 행렬곱은 여전히 큰 계산량
- Softmax 계층 계산 역시 동일 : k번째 원소를 target으로 했을때 우리가 구해야하는 값 y_k = exp(s_k) / sum(exp^(s_i)) : 결국 어휘수(예를들어 100만개라면 백만번)만큼 계산 필요.

### 4.2.2 다중 분류에서 이진 분류로
- 다중 분류를 이진 분류로 근사하는 것이 네거티브 샘플링을 이해하는 데 중요한 개념
- 즉 target 단어에 해당하는 index의 값만 확률로 구하는 것이 목표
  ![p4-2-2](https://user-images.githubusercontent.com/24144491/59848682-b60f0980-93a0-11e9-89a3-6d10f18e4059.png)

### 4.2.3 시그모이드 함수와 교차 엔트로피 오차
- 마지막 출력층 값을 확률로 바꾸기 위해 `sigmoid` 함수 활용 : y = 1 / (1 + exp(-x))
- ![s4-3](https://user-images.githubusercontent.com/24144491/59848684-b6a7a000-93a0-11e9-9760-a1d761ca31e0.png)

- [warning] 책에서는 [식 1.7] L = -(시그마_k(t_k x log(y_k)) 와 위의 [식 4.3]이 다중 분류에서 출력층에 뉴런을 2개만 사용할 경우 위의 식과 같아진다

- ![s4-4](https://user-images.githubusercontent.com/24144491/59848678-b5767300-93a0-11e9-934e-217c34371bd3.png)

> corrections : delta(L) / delta(y)  = - (t/y) + (1-t)/(1-y) = (y-t)/(y(1-y))

  : 출력층의 backward의 경우 L을 x(sigmoid 전)로 미분한 값, *y-t* 만 넘겨주면 된다. (매우 simple!)



### 4.2.4 다중 분류에서 이진 분류로 구현
- `class Embedding Dot`



### 4.2.5 네거티브 샘플링

- 4.2.4 까지는 target, 즉 정답인 단어에 해당하는 Loss만 구하게 된다. 그렇다면 정답이 아닌 다른 단어에 대한 확률값은 어떻게 구할지 잘 학습하지 못한다.
- 정답이 아닌 단어는 낮은 확률을 예측할 수 있게 학습하도록 부정적인 예, negative sample 몇 가지를 더 넣어주자.
- 예를 들어, you say goodbye and i hello . 에서 you 와 goodbye가 input으로 들어갔다면, 그에 대한 답은 [0, 1, 0, 0, ..  ,0] 이 되어야 한다. 따라서 target index, say의 index는 1로 예측해야하고, 나머지 index는 0으로 예측해야한다는 뜻. target = [0,1,0,..,0] 에서 정답 index는 1이고 나머지 0,2,3,4,5,...,vocab_size 는 모두 0으로 예측해야한다. 이때, 3,4,5 등을 부정적인 예로 추가해 출력층을 구성하고 그 출력층에 대한 loss를 계산하고 역전파한다면 몇 가지 부정적인 예시에 대해서도 잘 학습할 수 있다. 

### 4.2.6 네거티브 샘플링의 샘플링 기법
- 그렇다면 부정적인 예를 어떤 기준으로 sampling 할 것인가?
- 그에 대한 답은, 말뭉치의 단어별 출현 횟수를 바탕으로 확률 분포를 구한다. : `UnigramSampler(corpus, power, sample_size)`
- 이때, 출현 빈도가 낮은 단어의 선택을 높여주기 위해 확률 분포에서 구한 값들 0.75 제곱하고 해당 확률 값을 다시 구한다. 즉, 출현 빈도가 낮은 단어의 확률 값을 높여주고, 다른 확률 값은 상대적으로 낮출 수 있게 되어 비교적 골고루 단어가 선택되도록 하는 것이 목적이다.



### 4.2.7 네거티브 샘플링 구현

- `class NagetiveSamplingLoss`

```python

import collections
# 4.2.3 sigmoid with loss 구현
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  #  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

# 4.2.4 다중 분류에서 이진 분류로 구현
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh

# 4.2.7 네거티브 샘플링 구현
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = np.asnumpy(target[i])
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh

```

# 4.3 개선판 word2vec 학습

___



![p4-2-4](https://user-images.githubusercontent.com/24144491/59848683-b6a7a000-93a0-11e9-8b16-05a7180824f4.png)

- 4.3.1 CBOW 모델 구현
- 4.3.2 CBOW 모델 학습 코드
- 4.3.3 CBOW 모델 평가


```python
# 4.3.1 CBOW 모델 구현
import sys
sys.path.append('D:/ANACONDA/envs/tf-gpu/code/NLP')

class CBOW:
    def __init__(self, vocab_size,hidden_size,window_size, corpus):
        V, H = vocab_size, hidden_size
        
        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V,H).astype('f')
        W_out = 0.01 * np.random.rand(V,H).astype('f')
        
        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out,corpus, power=0.75, sample_size=5)
        
        # 모든 가중치와 기울기를 배열에 모은다.
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        #인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in
    
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:,i])
        h *= 1/ len(self.in_layers)
        loss = self.ns_loss.forward(h,target)
        return loss
    
    def backward(self, dout = 1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
    
# Skip-Gram
class SkipGram:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        rn = np.random.randn

        # 가중치 초기화
        W_in = 0.01 * rn(V, H).astype('f')
        W_out = 0.01 * rn(V, H).astype('f')

        # 계층 생성
        self.in_layer = Embedding(W_in)
        self.loss_layers = []
        for i in range(2 * window_size):
            layer = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
            self.loss_layers.append(layer)

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer] + self.loss_layers
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)

        loss = 0
        for i, layer in enumerate(self.loss_layers):
            loss += layer.forward(h, contexts[:, i])
        return loss

    def backward(self, dout=1):
        dh = 0
        for i, layer in enumerate(self.loss_layers):
            dh += layer.backward(dout)
        self.in_layer.backward(dh)
        return None
```


```python
# 학습을 위한 함수
import numpy
import time
import matplotlib.pyplot as plt
#from common.np import *  # import numpy as np
#from common.util import clip_grads
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=500):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # 뒤섞기¸°
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # ęłľě ë ę°ě¤ěšëĽź íëëĄ ëŞ¨ě
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('반복 (x'  + str(self.eval_interval) + ')')
        plt.ylabel('손실')
        plt.show()

#import sys
#sys.path.append('..')

def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
import pickle
#from common.trainer import Trainer
#from common.optimizer import Adam
class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

#from cbow import CBOW
#from common.util import create_contexts_target, to_cpu, to_gpu
def create_contexts_target(corpus, window_size=1):
    '''맥락과 타깃 생성

    :param corpus: 말뭉치(단어 ID 목록)
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return:
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)

```


```python
# 4.3.2 학습
from dataset import ptb

# 하이퍼 파라미터 설정
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# 데이터 읽기 + target, contexts 만들기
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)
    
# 모델 등 생성 - CBOW or SkipGram
model = CBOW(vocab_size, hidden_size, window_size, corpus)
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model,optimizer)

# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size, eval_interval = 3000) # eval_interval=500
trainer.plot()
```

    | 에폭 1 |  반복 1 / 9295 | 시간 0[s] | 손실 4.16
    | 에폭 1 |  반복 3001 / 9295 | 시간 26[s] | 손실 2.64
    | 에폭 1 |  반복 6001 / 9295 | 시간 52[s] | 손실 2.42
    | 에폭 1 |  반복 9001 / 9295 | 시간 78[s] | 손실 2.31
    | 에폭 2 |  반복 1 / 9295 | 시간 81[s] | 손실 2.27
    | 에폭 2 |  반복 3001 / 9295 | 시간 107[s] | 손실 2.20
    | 에폭 2 |  반복 6001 / 9295 | 시간 133[s] | 손실 2.14
    | 에폭 2 |  반복 9001 / 9295 | 시간 159[s] | 손실 2.09
    | 에폭 3 |  반복 1 / 9295 | 시간 162[s] | 손실 2.08
    | 에폭 3 |  반복 3001 / 9295 | 시간 188[s] | 손실 2.00
    | 에폭 3 |  반복 6001 / 9295 | 시간 214[s] | 손실 1.98
    | 에폭 3 |  반복 9001 / 9295 | 시간 240[s] | 손실 1.95
    | 에폭 4 |  반복 1 / 9295 | 시간 243[s] | 손실 1.94
    | 에폭 4 |  반복 3001 / 9295 | 시간 269[s] | 손실 1.87
    | 에폭 4 |  반복 6001 / 9295 | 시간 295[s] | 손실 1.87
    | 에폭 4 |  반복 9001 / 9295 | 시간 321[s] | 손실 1.86
    | 에폭 5 |  반복 1 / 9295 | 시간 324[s] | 손실 1.85
    | 에폭 5 |  반복 3001 / 9295 | 시간 350[s] | 손실 1.78
    | 에폭 5 |  반복 6001 / 9295 | 시간 376[s] | 손실 1.78
    | 에폭 5 |  반복 9001 / 9295 | 시간 402[s] | 손실 1.77
    | 에폭 6 |  반복 1 / 9295 | 시간 404[s] | 손실 1.78
    | 에폭 6 |  반복 3001 / 9295 | 시간 430[s] | 손실 1.70
    | 에폭 6 |  반복 6001 / 9295 | 시간 456[s] | 손실 1.71
    | 에폭 6 |  반복 9001 / 9295 | 시간 483[s] | 손실 1.71
    | 에폭 7 |  반복 1 / 9295 | 시간 485[s] | 손실 1.71
    | 에폭 7 |  반복 3001 / 9295 | 시간 511[s] | 손실 1.63
    | 에폭 7 |  반복 6001 / 9295 | 시간 538[s] | 손실 1.64
    | 에폭 7 |  반복 9001 / 9295 | 시간 564[s] | 손실 1.65
    | 에폭 8 |  반복 1 / 9295 | 시간 566[s] | 손실 1.65
    | 에폭 8 |  반복 3001 / 9295 | 시간 592[s] | 손실 1.58
    | 에폭 8 |  반복 6001 / 9295 | 시간 619[s] | 손실 1.59
    | 에폭 8 |  반복 9001 / 9295 | 시간 646[s] | 손실 1.59
    | 에폭 9 |  반복 1 / 9295 | 시간 648[s] | 손실 1.59
    | 에폭 9 |  반복 3001 / 9295 | 시간 675[s] | 손실 1.53
    | 에폭 9 |  반복 6001 / 9295 | 시간 701[s] | 손실 1.54
    | 에폭 9 |  반복 9001 / 9295 | 시간 727[s] | 손실 1.55
    | 에폭 10 |  반복 1 / 9295 | 시간 730[s] | 손실 1.56
    | 에폭 10 |  반복 3001 / 9295 | 시간 756[s] | 손실 1.48
    | 에폭 10 |  반복 6001 / 9295 | 시간 782[s] | 손실 1.49
    | 에폭 10 |  반복 9001 / 9295 | 시간 809[s] | 손실 1.50



![output_13_1](https://user-images.githubusercontent.com/24144491/59848685-b7d8cd00-93a0-11e9-8e79-650b5cf8d61c.png)



```python
# word_vecs, 분산표현 저장
word_vecs = model.word_vecs
if GPU :
    word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params1.pkl'
with open(pkl_file,'wb') as f:
    pickle.dump(params,f,-1);
```


```python
# 4.3.3 CBOW 모델 평가
# GPU -> CPU, cupy -> numpy 로
#from common.util import most_similar, analogy
import numpy as np
def cos_similarity(x, y, eps=1e-8):
    '''코사인 유사도 산출

    :param x: 벡터
    :param y: 벡터
    :param eps: '0으로 나누기'를 방지하기 위한 작은 값
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''유사 단어 검색

    :param query: 쿼리(텍스트)
    :param word_to_id: 단어에서 단어 ID로 변환하는 딕셔너리
    :param id_to_word: 단어 ID에서 단어로 변환하는 딕셔너리
    :param word_matrix: 단어 벡터를 정리한 행렬. 각 행에 해당 단어 벡터가 저장되어 있다고 가정한다.
    :param top: 상위 몇 개까지 출력할 지 지정
    '''
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
        
        
def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x

def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s(을)를 찾을 수 없습니다.' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


```


```python
pkl_file = './cbow_params1.pkl'
with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    
word_vecs = params['word_vecs']
word_to_id = params['word_to_id']
id_to_word = params['id_to_word']

# 가장 비슷한(most similar) 단어 뽑기
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
```


    [query] you
     we: 0.7529296875
     i: 0.724609375
     your: 0.623046875
     someone: 0.60693359375
     anybody: 0.60595703125
    
    [query] year
     month: 0.84033203125
     week: 0.76953125
     spring: 0.75634765625
     summer: 0.73681640625
     decade: 0.7021484375
    
    [query] car
     window: 0.61376953125
     truck: 0.59765625
     luxury: 0.59326171875
     auto: 0.58984375
     cars: 0.5625
    
    [query] toyota
     nissan: 0.6650390625
     nec: 0.64697265625
     honda: 0.64501953125
     minicomputers: 0.630859375
     ibm: 0.6279296875



```python
# 유추(analogy) 작업
print('-'*50)
analogy('king', 'man', 'queen',  word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs)
```

    --------------------------------------------------
    
    [analogy] king:man = queen:?
     a.m: 6.46875
     woman: 5.25390625
     father: 4.7265625
     daffynition: 4.70703125
     toxin: 4.61328125
    
    [analogy] take:took = go:?
     're: 4.33203125
     went: 4.1796875
     came: 4.17578125
     were: 3.966796875
     are: 3.89453125
    
    [analogy] car:cars = child:?
     a.m: 6.93359375
     rape: 5.96484375
     daffynition: 5.8046875
     children: 5.3125
     incest: 5.3046875
    
    [analogy] good:better = bad:?
     more: 5.72265625
     less: 5.453125
     rather: 5.37890625
     greater: 4.5703125
     faster: 4.2265625




# 4.5 정리

___

- Embedding 계층은 단어의 분산 표현을 담고 있다
- word2vec의 개선을 위해 다음 2가지 작업을 수행했다.
  - Embedding 계층에서 특정 단어의 index만 뽑아 계산하도록
  - Negative sampling을 통해 다중 분류를 이진 분류로, 몇 가지의 단어들의 확률값과 Loss를 계산하도록
   - wod2vec의 Embedding, 분산 표현에는 단어의 의미가 들어가 있고 비슷한 맥락에서 사용되는 단어는 Embedding 공간에서 서로 가까이 위치한다.
- word2vec의 Embedding, 분산 표현을 이용하면 유추 문제를 벡터의 덧셈과 뺄셈 문제로 풀 수 있다.
- 전이 학습 측면에서 특히 중요하며, 단어의 분산 표현은 다양한 자연어 처리 작업에 이용할 수 있다.

- 마지막으로 CBOW 도식화를 그려보며 정리해 보았다.

![p4-5](https://user-images.githubusercontent.com/24144491/59949997-b2b67380-94af-11e9-8dba-de676b304c5f.png)





# 4.6 번외

___

- [1] 사실 위의 그림은 간단히 설명하기 위해서 저렇게 나타냈지만, 책에서 실제로 구현된 코드는 위의 그림과는 다르다. Class Embedding인 W_in과 W_out을 각각 [window_size * 2] 와 [Target + Sample_size]만큼 만들고 forward와 backward를 진행하고 중복된 가중치를 하나로 다시 모으는 작업을 `remove_duplicate`로 처리한다. 이 과정이 번거롭다고 느껴 W_in을 1개로 통합하고, Target과 Sample_size 용을 위한 W_out을 2개 마련해 Weight를 줄여 학습을 시도해 보았다.
  - 위의 과정을 위해서는 다음 Class의 __init__ , forward 와 backward 부분을 고쳐야 한다
    - CBOW
    - NegativeSamplingLoss
    - Embedding
  - 고친 코드는 [ch4_word2vec_faster_improved](https://github.com/Taeu/NLP/blob/master/ch4_word2vec_faster_improved.ipynb) 에 올려두었다.
  - 속도는 교재보다 2.3배 빠르지만 성능은 더 느리게 개선되며 같은 시간을 학습했을 때 교재의 코드가 더 높은 accuracy를 보였다.
    - 그 이유는 내가 꼼꼼하게 본다고 봤지만 아주 세심히 다루지 못했을 가능성이 크고
    - 혹은 W_in 을 backward 할 때 여러 뉴런에 해당할 수 있는 weight들을 len(self.in_layers)로 나눠주면서 더 미세하게 조정해주어서이지 않을까?
- [2] `Skip Gram`모델로도 돌려 보았다. 코드는 [ch4_word2vec_TU](https://github.com/Taeu/NLP/blob/master/ch4_word2vec_TU.ipynb) 에 업로드했다. 
  - CBOW 보다 학습이 훨씬 오래 걸린다. (약 6배 차이)
  - 유추 문제의 정확도는 CBOW를 돌렸을 때 보다 더 떨어졌다.
- [ch4_word2vec_faster](https://github.com/Taeu/NLP/blob/master/ch4_word2vec_faster.ipynb)

