---

layout: post
title: "[논문] VAE(Auto-Encoding Variational Bayes) 직관적 이해"
category: paper
tags: dl paper generativeModel unsupervised
comments: true
img: dimension2.jpg

---


# VAE : Variational Auto-Encoder를 직관적으로 이해하기!

 최근 딥러닝 스터디 중 한 군데에서 케라스 창시자에게 배우는 딥러닝 교재로 공부하고 있다. 마지막 8장 부분은 일반적인 딥러닝의 여러 부분을 커버하고 있다. 

- 8.1 Text Generation with LSTM
- 8.2 Implementing DeepDream
- 8.3 Performing Neural Style Transfer
- 8.4 Variatinoal Autoencoder(VAE)
- 8.5 Generative Adversarial Networks(GAN)

 이 중에서 이번주에는 8.4, 8.5를 다루는데 8.4 VAE 부분을 발표하기로 했다. 일단 케라스 8.4를 읽고 코드를 돌려보면서 대략적인 흐름은 파악했다. 하지만 encoder부분이 의마하는게 무엇인지, 왜 encoder 결과가 mu와 var인지 그리고 decoder은 어떤 의미를 가지며, 각각의 loss는 어떻게 도출되었는지 제대로 설명이 나와있지 않았다. 
 
 그 이유를 찾고자 논문에 대한 내용을 읽어봤다. Abstract부터 막히고 관련 수식 전개에서 또 막혔다... 관련된 용어와 배경 지식들, Bayesian Prob.부터 여러 확률 통계 지식들을 구글링해가며 이해해 나갔다. 그 과정에서 오토인코더를 정말 잘 정리한 강의를 찾았는데 그것이 아래 참고자료에도 나와있지만 `이활석님의 오토인코더의 모든 것`의 강의다. 너무나도 꼼꼼히, 직관적으로 설명을 잘해놓으셨다. 꼭 해당 강의를 보길 강추한다. 
 
 수식이 많다보니까 수식을 한꺼번에 정리하면서 구조를 이해한 글을 적자니 너무 길어질 것 같아 이번글에서는 최대한 직관적으로 이해할 수 있게 수식은 최대한 줄이고 말로 풀어 써보았다.


## 참고자료
- [논문](https://arxiv.org/pdf/1312.6114.pdf)
- [네이버 이활석님의 슬라이드노트](https://www.slideshare.net/NaverEngineering/ss-96581209)
- [네이버 이활석님의 오토인코더의 모든 것 강의 in Youtube](https://www.youtube.com/watch?v=rNh2CrTFpm4&t=2206s)
- [이웅원님의 VAE 관련 tutorial 글](https://dnddnjs.github.io/paper/2018/06/19/vae/)
- [코드 in Github : Deeplearning-with-Python-케라스 딥러닝 교재 원서](https://github.com/Taeu/FirstYear_RealWorld/blob/master/GoogleStudy/Keras_week8_2/8.4%20VAE.ipynb)


# Domain Gap 
___

**VAE**는 논문을 이해하려면 꽤 많은(적어도 나에게는) 사전지식이 필요하다. 간단하게 정리하면 다음과 같다. (자세한 설명은 참고링크를 확인하기 바란다.)

> [1] VAE는 Generative Model이다.

- **Generative Model**이란 training data가 주어졌을 때 이 training data가 가지는 real 분포와 같은 분포에서 sampling된 값으로 new data를 생성하는 model을 말한다.
- [이웅원님의 Generative model에 관한 설명글](https://dnddnjs.github.io/paper/2018/06/19/vae/)

> [2] 확률 통계 이론 (Bayseain, conditional prob, pdf etc)

- **베이지안 확률(Bayesian probability)**: 세상에 반복할 수 없는 혹은 알 수 없는 확률들, 즉 일어나지 않은 일에 대한 확률을 사건과 관련이 있는 여러 확률들을 이용해 우리가 알고싶은 사건을 추정하는 것이 베이지안 확률이다.
- [베이지안 이론 관련 설명글](http://bioinformaticsandme.tistory.com/47) 

> [3] 관련 용어들 

- **latent** : '잠재하는', '숨어있는', 'hidden'의 뜻을 가진 단어. 여기서 말하는 latent variable z는 특징(feature)를 가진 vector로 이해하면 좋다.
- **intractable** : 문제를 해결하기 위해 필요한 시간이 문제의 크기에 따라 지수적으로 (exponential) 증가한다면 그 문제는 난해 (intractable) 하다고 한다.
- **explicit density model** : 샘플링 모델의 구조(분포)를 명확히 정의
- **implicit density model** : 샘플링 모델의 구조(분포)를 explicit하게 정의하지 않음
- **density estimation** : x라는 데이터만 관찰할 수 있을 때, 관찰할 수 없는 x가 샘플된 확률밀도함수(probability density function)을 estimate하는 것
- **Gaussian distribution** : 정규분포
- **Bernoulli distribution** : 베르누이분포
- **Marginal Probability** : 주변 확률 분포
- **D_kl** : 쿨백-라이블러 발산(Kullback–Leibler divergence, KLD), 두 확률분포의 차이
- **Encode / Decode**: 암호화,부호화 / 암호화해제,부호화해제
- **likelihood** : **가능도. 이에 대한 설명은 꼭 아래 링크에 들어가 읽어보길 바란다.**
- [likellihood에 대한 설명글](http://rpubs.com/Statdoc/204928)

등의 개념들을 숙지하고 넘어가야 논문에 대한 내용을 이해를 할 수 있다.

> 번외) [4] Auto-Encoder

- VAE와 오토인코더(AE)는 목적이 전혀 다르다.
- 오토인코더의 목적은 어떤 데이터를 잘 압축하는것, 어떤 데이터의 특징을 잘 뽑는 것, 어떤 데이터의 차원을 잘 줄이는 것이다.
- 반면 VAE의 목적은 Generative model으로 어떤 새로운 X를 만들어내는 것이다.

# VAE
___

이제부터 본격적으로 VAE 관련된 내용들을 코드와 함께 살펴보자. 기존의 논문의 흐름은 Generative Model이 가지는 문제점들을 해소하기 위해 어떤 방식을 도입했는지 차례차례 설명하고있다. 하지만 관련된 수식도 많고 중간에 생략된 식도 많아 논문대로 따라가다보면 전체적인 구조를 이해하기 힘들기때문에 먼저 구조를 살펴본 뒤 각 구조가 가지는 의미가 무엇인지(왜 이 구조가 나왔는지) 살펴보고 최종적으로 정리하도록 할 예정이다. (Top-down approach(?))

## **VAE GOAL** 
___

논문 Abstract에 나와있는 첫 문장이다. **이 목적을 이해하는 것이 가장 중요하니 천천히 보면서 이해하기 바란다.**

> How can we perform efficient inference and learning in directed probabilistic
models, in the presence of continuous latent variables with intractable posterior
distributions, and large datasets? 

![explain](https://user-images.githubusercontent.com/24144491/50323471-1968cd80-051d-11e9-86d9-5d7f90519e5d.png)


VAE의 목표는 `Generative Model`의 목표와 같다. (1) data와 같은 분포를 가지는 sample 분포에서 sample을 뽑고(2) 어떤 새로운 것을 생성해내는 것이 목표다. 즉,

- (1) 주어진 training data가 p_data(x)(확률밀도함수)가 어떤 분포를 가지고 있다면, sample 모델 p_model(x) 역시 같은 분포를 가지면서, (sampling 부분)
- (2) 그 모델을 통해 나온 inference 값이 새로운 x라는 데이터이길 바란다. (Generation 부분)

예를 들어, 몇 개의 다이아몬드(training data)를 가지고 있다고 생각해보자. 그러면 training 다이아몬드 뿐만아니라 모든 **다이아몬드**의 확률분포와 똑같은 분포를 가진 모델에서 값을 뽑아(1. sampling) training 시켰던 다이아몬드와는 다른 또 다른 다이아몬드(new)를 만드는(generate) 것이다.


## **VAE 구조**
___
백문이 불어일견. VAE의 전체 구조를 한 도식으로 살펴보자.

![arch1](https://user-images.githubusercontent.com/24144491/50323466-18d03700-051d-11e9-82ed-afb1b6e2666a.png)

케라스 교재에 구현된 코드와 논문의 구조는 약간의 차이가 있다. 전체적인 구조는 똑같으니 크게 헷갈릴 것은 없지만, 그래도 코드의 약간의 변형된 부분은 다음과 같다. 

> 논문과 다른점 :  Input shape, Encoder의 NN 모델, Decoder의 NN모델 (코드에서는 왼쪽의 각 부분들을 DNN을 CNN구조로 바꿈)

위의 도식은 VAE 구조를 완벽히 정리한 그림이다. 이제 이 그림을 보면서, input 그림이 있을 때 어떤 의미를 가진 구조를 거쳐 output이 나오게 되는지 3 단계로 나누어 살펴보자.

1. input: x --> 𝑞_∅ (𝑥)-->  𝜇_𝑖,𝜎_𝑖
2. 𝜇_𝑖, 𝜎_𝑖, 𝜖_𝑖 -->  𝑧_𝑖
3. 𝑧_𝑖 --> 𝑔_𝜃 (𝑧_𝑖) -->  𝑝_𝑖 : output

### 1. Encoder


input: x --> 𝑞_∅ (𝑥)-->  𝜇_𝑖,𝜎_𝑖


![arch2](https://user-images.githubusercontent.com/24144491/50323467-1968cd80-051d-11e9-932c-a9b9ef91de58.png)


```python
img_shape = (28,28,1)
batch_size = 16
latent_dim = 2

input_img = keras.Input(shape = img_shape)
x = layers.Conv2D(32,3,padding='same',activation='relu')(input_img)
x = layers.Conv2D(64,3,padding='same',activation='relu',strides=(2,2))(x)
x = layers.Conv2D(64,3,padding='same',activation='relu')(x)
x = layers.Conv2D(64,3,padding='same',activation='relu')(x)

shape_before_flattening = K.int_shape(x) # return tuple of integers of shape of x

x = layers.Flatten()(x)
x = layers.Dense(32,activation='relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
```


- Input shape(x) : (28,28,1)
- 𝑞_∅ (𝑥) 는 encoder 함수인데, x가 주어졌을때(given) z값의 분포의 평균과 분산을 아웃풋으로 내는 함수이다.
- 다시말해 q 함수(=Encoder)의 output은 𝜇_𝑖,𝜎_𝑖 이다.

어떤 **X**라는 입력을 넣어 인코더의 아웃풋은 𝜇_𝑖,𝜎_𝑖 이다. 어떤 데이터의 특징을(latent variable) X를 통해 추측한다. 기본적으로 여기서 나온 특징들의 분포는 정규분포를 따른다고 가정한다. 이런 특징들이 가지는 확률 분포 𝑞_∅ (𝑥) (정확히 말하면 $$$q_{\phi}z\vert(x^{(i)})$$$의 true 분포 (= $$$p_{\theta}(x^{(i)} \vert z)$$$)를 정규분포(=Gaussian)라 가정한다는 말이다. 따라서 latent space의 latent variable 값들은 𝑞_∅ (𝑥)의 true 분포를 approximate하는 𝜇_𝑖,𝜎_𝑖를 나타낸다.

> Encoder 함수의 output은 latent variable의 분포의 𝜇 와 𝜎 를 내고, 이 output값을 표현하는 확률밀도함수를 생각해볼 수 있다.


### 2. Reparameterization Trick (Sampling)


𝜇_𝑖, 𝜎_𝑖, 𝜖_𝑖 -->  𝑧_𝑖


![arch3](https://user-images.githubusercontent.com/24144491/50323468-1968cd80-051d-11e9-8031-ff19c4eaf7c8.png)


```python
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dim),mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])
```

만약 Encoder 결과에서 나온 $$𝜇_𝑖,𝜎_𝑖$$ 값을 활용해 decoding 하는데 sampling 하지 않는다면 어떤 일이 벌어질까? 당연히 $$𝜇_𝑖,𝜎_𝑖$$는 한 값을 가지므로 그에 대한 decoder(NN)역시 한 값만 뱉는다. 그렇게 된다면 어떤 한 variable은 무조건 똑같은 한 값의 output을 가지게 된다. 

하지만 Generative Model, VAE가 하고 싶은 것은, 어떤 data의 true 분포가 있으면 그 분포에서 하나를 뽑아 기존 DB에 있지 않은 새로운 data를 생성하고 싶다. 따라서 우리는 필연적으로 그 데이터의 확률분포와 같은 분포에서 하나를 뽑는 sampling을 해야한다. 하지만 그냥 sampling 한다면 sampling 한 값들을 backpropagation 할 수 없다.(아래의 그림을 보면 직관적으로 이해할 수 있다) 이를 해결하기 위해 reparmeterization trick을 사용한다. 

![reparameters](https://user-images.githubusercontent.com/24144491/50323463-18d03700-051d-11e9-8109-d0b4530d77b4.PNG)


정규분포에서 z1를 샘플링하는 것이나, 입실론을 정규분포(자세히는 N(0,1))에서 샘플링하고 그 값을 분산과 곱하고 평균을 더해 z2를 만들거나 두 z1,z2 는 같은 분포를 가지기 때문이다. 그래서 코드에서 `epsilon`을 먼저 정규분포에서 random하게 뽑고, 그 epsilon을 exp(z_log_var)과 곱하고 z_mean을 더한다. 그렇게 형성된 값이 z가 된다.

> latent variable에서 sample된 z라는 value (= decoder input)이 만들어진다.

### 3. Decoder

𝑧_𝑖 --> 𝑔_𝜃 (𝑧_𝑖) -->  𝑝_𝑖 : output


![arch4](https://user-images.githubusercontent.com/24144491/50323470-1968cd80-051d-11e9-90b1-a2e69ddbcbdd.png)

```python
## 8.25 VAE decoder network, mapping latent space points to imgaes

decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]),activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32,3,padding='same',activation='relu',strides=(2,2))(x)
x = layers.Conv2D(1,3,padding='same',activation='sigmoid')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)

```

z 값을 g 함수(decoder)에 넣고 deconv(코드에서는 Conv2DTranspose)를 해 원래 이미지 사이즈의 아웃풋 z_decoded가 나오게 된다. 이때 p_data(x)의 분포를 **Bernoulli** 로 가정했으므로(이미지 recognition 에서 Gaussian 으로 가정할때보다 Bernoulli로 가정해야 의미상 그리고 결과상 더 적절했기 때문) output 값은 0~1 사이 값을 가져야하고, 이를 위해 activatino function을 sigmoid로 설정해주었다. (Gaussian 분포를 따른다고 가정하고 푼다면 아래 loss를 다르게 설정해야한다.)



## VAE 학습
___

### Loss Fucntion 이해
Loss 는 크게 총 2가지 부분이 있다.

![loss](https://user-images.githubusercontent.com/24144491/50323472-1a016400-051d-11e9-86b7-d8bf6a1a880f.png)

```python
 def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x,z_decoded)
        kl_loss   = -5e-4*K.mean(1+z_log_var-K.square(z_mean)-K.exp(z_log_var),axis=-1)
        return K.mean(xent_loss + kl_loss)

```

- **Reconstruction Loss**(code에서는 xent_loss)
- **Regularization Loss**(code에서는 kl_loss)

일단 직관적으로 이해를 하자면, 

1. Generative 모델답게 새로운 X를 만들어야하므로 X와 만들어진 output, New X와의 관계를 살펴봐야하고, 이를 **Reconstruction Loss** 부분이라고 한다. 이때 디코더 부분의 pdf는 Bernoulli 분포를 따른다고 가정했으므로 그 둘간의 **cross entropy**를 구한다( 이 부분에 대해서 왜 같은지는 수식을 포함한 포스터에서 더 상세히 다룰 것이다)

2. X가 원래 가지는 분포와 동일한 분포를 가지게 학습하게 하기위해 true 분포를 approximate 한 함수의 분포에 대한 loss term이 **Regularization Loss**다. 이때 loss는 true pdf 와 approximated pdf간의 **D_kl(두 확률분포의 차이(거리))**을 계산한다. (이 부분도 역시 왜 이런 식이 나왔는지는 수식을 포함한 포스텅서 더 상세히 다룰 것이다)


### 학습

encoder 부분과 decoder 부분을 합쳐 한 모델을 만들고 train 하면 끝! 자세한 코드는 [Github](https://github.com/Taeu/FirstYear_RealWorld/blob/master/GoogleStudy/Keras_week8_2/8.4%20VAE.ipynb)에 올려두었으니 참고하기 바란다.

## VAE 결과
___

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

n=20
digit_size = 28
figure = np.zeros((digit_size*n,digit_size*n))
grid_x = norm.ppf(np.linspace(0.05,0.95,n))
grid_y = norm.ppf(np.linspace(0.05,0.95,n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi,yi]])
        z_sample = np.tile(z_sample,batch_size).reshape(batch_size,2)
        x_decoded = decoder.predict(z_sample, batch_size = batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i+1)*digit_size, j*digit_size:(j+1)*digit_size] = digit
        
plt.figure(figsize=(10,10))
plt.imshow(figure, cmap ='Greys_r')
plt.show()
```

![manifold2](https://user-images.githubusercontent.com/24144491/50323704-289c4b00-051e-11e9-8dad-f32bad3c0583.PNG)

위의 코드를 실행시키면 위 그림에서 오른쪽과 같은 도식이 나오는데 학습이 잘 되었다면 차원의 manifold를 잘 학습했다는 말이다. 그 manifold를 2차원으로 축소시킨 것(z1,z2)에서 z1 20개(0.05~0.95), z2 20개, 총 400개의 순서쌍의 xi,yi에서 sample을 뽑아 시각화한것이 오른쪽 그림인데 2D상에서 거리의 유의미한 차이에 따라 숫자들이 달라지는 것을 확인할 수 있으며, 각 숫자 상에서도 서로 다른 rotation들을 가지고 있다는 것이 보인다.


## Insight

마지막으로 VAE는 왜하냐고 물어본다면 크게 2가지로 답할 수 있다.

>1. **Generative Model 목적 달성**
2. **Latent variable control 가능**

- Generative Model을 통해 적은 data를 가지고 원래 data가 가지는 분포를 꽤 가깝게 근사하고 이를 통해 새로운 data를 생성해낼 수 있다는 점.
- Latent variable의 분포를 가정해 우리가 sampling 할 값들의 분포를 control 할 수 있게 되고, manifold도 잘 학습이 된다는점. 
- 이는 data의 특징들도 잘 알 수 있고, 그 특징들의 분포들은 크게 벗어나지 않게 control 하면서 그 속에서 새로운 값을 만들 수 있다는 점.

정도가 될 것 같다.

> VAE의 한계점을 극복하기 위해, CVAE, AAE가 나왔으니 관심있는 사람은 관련된 자료를 찾아보기 바란다


## 끝으로

한 주 동안 내가 잘 몰랐던 Generative 모델들 그리고 관련 배경지식, domain gap을 줄이면서 이해하고 발표하는 글을 쓰려고 하다보니 찾아볼 것도 많았고, 이해하기 위해 생각도 그만큼 많이 했다. 일, 운동, 기타 자기계발 활동을 하는 것 외에 공부는 온전히 이 부분만 했다. 그래도 내가 모르는 부분을 알게되어 기쁘면서 한 편으로는 unsupervised 쪽도 필히 공부해야겠다는 마음이 들어 더 많은 숙제를 받은 것 같아 마음이 무겁기도 하다. 그리고 수식 정리 글도 금요일에 일 끝나고 밤에 정리한다고 했는데 다 못했고(이해가 확실히 안되는 부분이 약간 있어서..)그 글을 정리하다보니 이 글을 다시 수정해야 할 필요가 있어 이것을 수정하는 것으로 발표준비는 마무리. 이번 주는 꽤 바쁘게 살면서도 하루 routine을 균형잡게 짜 놓아서 그런지 스트레스 없이 뿌듯함을 느낄 수 있는 한 주였다. (사실 이런 회고의 글 쓸 시간에 수식 정리 글을 마무리하면 참 좋으련만ㅎㅎ)


