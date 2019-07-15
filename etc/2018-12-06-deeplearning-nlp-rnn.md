---
layout: post
title: "[DL] RNN 이해, 용어 정리" 
category: nlp
tags: dl nlp rnn
comments: true
img: nlp2.png
---

# RNN 관련 그림, 용어 정리
___

**Deep Learning with Python**(by Francois Chollet, 번역서는 케라스 창시자에게 배우는 딥러닝)으로 스터디 하다가 6장 NLP 부분에서 Keras의 RNN이 어떻게 구현돼 있는지 보다가, `units`라는 것이 어떤 의미를 가지는지 이해하는데 어려움을 겪었다. 그래서 오늘은 그 unit이 무엇인지를 조금 명확히 한 후, 여러가지 NLP 그림들이 있는데 여러 그림들을 보다 더 잘 이해할 수 있게 구글에 'RNN' 혹은 'LSTM'하면 검색해서 나오는 그림들의 차이점들을 분석해보고자 한다.

# RNN 이해하기
___

**`RNN`**은 **Recurrent** + **Neural Network**이다. 뉴럴 네트워크이긴한데 구조가 재귀적이라는 말. 다시 말해 어떤 **상태(state)** 혹은 한 함수에서의 상황, 상태에서 나온 결과값이 다음 상태 혹은 다음 함수의 input, 인자로 들어가게 된다는 말이다. 역시 말보다는 코드와 그림이 더 편할 것이기에 그림과 위에서 언급한 책에서 가져와봤다.

![2](https://user-images.githubusercontent.com/24144491/49584600-363ec600-f99f-11e8-909d-e1b93b85b89e.png)

구글에 검색을 하면 총 2가지 형태의 그림이 나오는데 선뜻보면, 아 이게 두개가 똑같고 색칠정도만 다르구나라고 생각할 수 있는데 **`절대 다른 그림이라는 것을 명시`**해야한다. 보다 RNN 자체에 대한 설명과 후에 나오는 Keras의 RNN 코드를 이해하기 위해서는 왼쪽그림이 더욱 맞는 표현이라고 판단하여 왼쪽그림을 가지고 이해한 뒤, 추가적인 레이어를 더 붙인 형태인 오른쪽 그림도 살펴볼 예정.

![3](https://user-images.githubusercontent.com/24144491/49584601-363ec600-f99f-11e8-822f-ff4c2a80e2a2.png)

```python
# Listing 6.19 Pseudocode RNN

state_t = 0 # state_t 를 0(vector)으로 초기화
for input_t in input_sequence:
	output_t = f(input_t, state_t)
    state_t = output_t
```

RNN이 돌아가는 구조를 위의 그림과 코드를 보고 큰 그림을 생각해보자. 그 전에 코드와 그림에서 용어 정리부터 하겠다.
- **`state_t`** :  time = t일 때 RNN을 통해 나온 결과값, 상태값을 의미한다. 위의 그림에서는 **`h_t`**와 같다.
- **`input_t`** : time = t일 때의 입력 값. 위의 그림에서는 **`x_t`**와 같다.
- **state_t**와 **input_t**, 그림에서 **h_t**와 **x_t**는 모두 **`vector`** 형태이다. 단순히 1, 2라는 숫자값이 아님.
- **`input_sequence`** : 인풋의 시퀀스, time step과 같은 개념이라고 생각하면 된다. 그림에서는 총 t+1개의 sequence가 있다. 따라서 원래 input은 (input_sequences, input_features)

즉, for문을 한 번 돌때마다, input_t와 state_t를 input으로 하는 `f`라는 function이 실행되고 그 output_t는 다시 state_t가 된다. 따라서 다음 state에서는 이전 state에서의 결과값이 다음 state에서 input이 되기때문에 recurrent 구조를 가지게 된다.



# NUMPY로 RNN 구현하기
___



```python
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features)) # input data

state_t = np.zeros(output_features,)) # initial state : an all-zero vector, (64,)

W = np.random.random((output_features, input_features))  # (64,32)
U = np.random.random((output_features, output_features)) # (64,64)
b = np.random.random((output_features,)) # (64,)

succeesive_outputs = []

# RNN 수행
for input_t in inputs:
	output_t = np.tanh(np.dot(W,input_t) + np.dot(U,state_t) + b)
    	successive_outputs.append(output_t)
    	state_t = output_t

final_output_sequences = np.concatenate(successive_outputs, axis = 0) 
# 위의 final_output_sequnce의 shape = (100, 64) = (timesteps, output_features)
```

이 그림을 말로 설명하는 것보다 그림으로 표현하는게 더 이해하기 수월할 것이라고 생각하여 도식화 해 보았다.

![5](https://user-images.githubusercontent.com/24144491/49584602-36d75c80-f99f-11e8-801e-c790f1204ff7.png)

이걸 제일 처음 봤던 그림이랑 비교해서 그려보자면 다음과 같다.

![3](https://user-images.githubusercontent.com/24144491/49584601-363ec600-f99f-11e8-822f-ff4c2a80e2a2.png)

![7](https://user-images.githubusercontent.com/24144491/49584604-36d75c80-f99f-11e8-92fb-d9d66de2bd99.png)


# RNN 그림 이해하기 (두 그림의 차이점)
___

![2](https://user-images.githubusercontent.com/24144491/49584600-363ec600-f99f-11e8-909d-e1b93b85b89e.png)

다시 돌아와서 이 그림을 본다면 차이점이 조금씩 눈에 들어오기 시작할 것이다. 분명한 차이점은 다음과 같다.

![default](https://user-images.githubusercontent.com/24144491/49584621-3dfe6a80-f99f-11e8-8d7a-f234d869129d.png)

![edit1](https://user-images.githubusercontent.com/24144491/49588920-5d9b9000-f9ab-11e8-8e2c-6043aa1ef403.png)

노란색 빗금친 영역이 바로 오른쪽 그림을 포괄하는 것. 여기서 왼쪽 그림에 추가된 점은 

> **`O`** 라는 노드와 그에 연결된 **`V`**라는 벡터

다시 말해, 왼쪽과 오른쪽의 차이점은, 왼쪽 그림은 오른쪽 그림을 포함함과 동시에 o라는 새로운 output을 만들어내고 있다는 점이다. 왼쪽 부분을 설명한 그림은 아래와 같다.

![default](https://user-images.githubusercontent.com/24144491/49584609-3a6ae380-f99f-11e8-89d3-9a8e3dc86fc1.png)


> 따라서 우리가 앞서 살펴본 `Numpy로 RNN 구현하기` 예제에서는 왼쪽그림이 아닌 오른쪽 그림으로 이해해야한다.

```python
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features)) # input data

state_t = np.zeros(output_features,)) # initial state : an all-zero vector, (64,)

W = np.random.random((output_features, input_features))  # (64,32)
U = np.random.random((output_features, output_features)) # (64,64)
b = np.random.random((output_features,)) # (64,)

succeesive_outputs = []

# RNN 수행
for input_t in inputs:
	output_t = np.tanh(np.dot(W,input_t) + np.dot(U,state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

final_output_sequences = np.concatenate(successive_outputs, axis = 0) 
# 위의 final_output_sequnce의 shape = (100, 64) = (timesteps, output_features)
```

다음의 코드를 도식화하면 다음과 같다.


![code1](https://user-images.githubusercontent.com/24144491/49585113-d3e6c500-f9a0-11e8-8aa9-ec9de92fce53.png)






# Keras로 RNN 짜보기
___

```python
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()
```

![1](https://user-images.githubusercontent.com/24144491/49584618-3dfe6a80-f99f-11e8-9c81-131478217946.PNG)

이 모델을 도식화하면 다음과 같다

![2](https://user-images.githubusercontent.com/24144491/49584624-3e970100-f99f-11e8-9937-1cc5794c6ea2.png)


- **`x`** : 임베딩 아웃풋의 단어 임베딩 값 [0, 0, .. , 0] (32,)
- **`h`** : SimpleRNN의 output (ht 혹은 st) [0, 0, ... , 0] (32,)
- **`weight`** : W, U, bias
- **`W`** : (output_features, input_features) = (32,32)
- **`U`** : (output_features, output_features) = (32,32)
- **`b`** : (output_features,) = (32,)
- **`num(#) of parameters`** : W + U + b = 32 x 32 + 32 x 32 + 32 = 32 x 65 = 2080 = output_featrues x (output_features + input_features + 1(bias))


```python
from keras import backend as K

K.clear_session()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()
```

> 전체 시퀀스를 다 얻으려면 위와 같이 return_sequences = True 로 바꿔주자.

![2](https://user-images.githubusercontent.com/24144491/49584619-3dfe6a80-f99f-11e8-9d7d-cb28a1b892d3.PNG)



![3](https://user-images.githubusercontent.com/24144491/49584625-3f2f9780-f99f-11e8-8f2b-f88785288d9d.png)


```python
from keras.layers import Dense

K.clear_session()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
```

위의 코드를 도식화하면 다음과 같다.

![4](https://user-images.githubusercontent.com/24144491/49584616-3d65d400-f99f-11e8-8a94-ce2c05d299f3.png)



## 용어 정리
___

추가로 [Keras Documentation : RNN](https://keras.io/layers/recurrent/) 부분을 읽을때 오해할 만한 용어들에 대해 정리해보고자 한다.

>- cell
- unit
- h_t
- output_t(위의 numpy로 구현하는 RNN 예제)
- state_t (keras doc : RNN)
- previous_output

들은 모두 같은 의미이다. 즉 이들은 모두 RNN계열의 모델들이 **`output vector`**를 의미한다. 한 덩어리라고 생각하면 된다. 모두 하나의 **`vector`**이다.

>- num(#) of units
- dimensionality of the output space
- h_t 의 dimension
- state_t의 dimension (keras doc : RNN)
- units (keras doc : RNN, LSTM, GRU etc)
- output_features (위에 numpy로 구현하는 RNN 예제에서)
- RNN(32), LSTM(32), GRU(32) 에서의 32라는 input

들은 모두 같은 말이다. 즉 이 용어들은 **`dimension`**을 의미한다. 모두 하나의 **`숫자`**로 표현될 수 있다.

> 참고로 output이라는 것은 어떤 layer 까지 고려한지에 따라 달라질 수 있다. 내가 언급한 output이란 keras에서 RNN, LSTM, GRU 등의 한 state를 돌고나서의 출력값들을 의미한다. 

이 사항들을 명확히하고 RNN 계열의 문서나 그림이나 코드들을 본다면 이제 제대로 이해할 수 있을 것이다. 

